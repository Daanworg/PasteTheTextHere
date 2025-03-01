"""
RAG Extension module for PasteTheTextHere

This module provides a Retrieval-Augmented Generation (RAG) system that can be integrated
with the existing PasteTheTextHere application. It demonstrates how to implement RAG
using Hugging Face tools and models.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F

class RAGSystem:
    """A Retrieval-Augmented Generation system for text processing."""
    
    def __init__(self, api_key=None):
        """Initialize the RAG system.
        
        Args:
            api_key: Hugging Face API key (uses environment variable if not provided)
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
            
        # Models for embeddings and generation
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.generation_model_name = "facebook/bart-large-cnn"  # Better for RAG-based generation
        
        # Chunk parameters
        self.chunk_size = 300  # words
        self.chunk_overlap = 50  # words
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load the necessary models."""
        logging.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, token=self.api_key)
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name, token=self.api_key)
        
        logging.info(f"Loading generation model: {self.generation_model_name}")
        self.generation_tokenizer = AutoTokenizer.from_pretrained(self.generation_model_name, token=self.api_key)
        
        # Note: We don't preload the generation model here to save memory
        # It will be loaded on-demand when needed in answer_question()
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            A list of text chunks
        """
        # Split text into words
        words = re.findall(r'\b\w+\b', text)
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A numpy array containing the embedding
        """
        # Tokenize and get model outputs
        inputs = self.embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            
        # Use mean pooling to get sentence embedding
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embedding = (sum_embeddings / sum_mask).squeeze().numpy()
        return embedding
    
    def process_document(self, document: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a document for RAG.
        
        Args:
            document: The document text to process
            metadata: Optional metadata about the document
            
        Returns:
            A dictionary containing the processed document data
        """
        # Split document into chunks
        chunks = self.chunk_text(document)
        
        # Generate embeddings for each chunk
        chunk_data = []
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            
            # Extract keywords (simple method)
            words = re.findall(r'\b\w+\b', chunk.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            keywords = [k for k, _ in keywords]
            
            # Store chunk data
            chunk_data.append({
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding.tolist(),
                "keywords": keywords,
                "metadata": metadata or {}
            })
            
        return {
            "chunks": chunk_data,
            "total_chunks": len(chunks),
            "document_metadata": metadata or {}
        }
    
    def retrieve_relevant_chunks(self, query: str, dataset: Dataset, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a query.
        
        Args:
            query: The query text
            dataset: The dataset containing chunks and embeddings
            top_k: Number of chunks to retrieve
            
        Returns:
            List of the most relevant chunks
        """
        # Generate embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Calculate similarity with all chunks in the dataset
        similarities = []
        for i, record in enumerate(dataset):
            chunk_embedding = torch.tensor(record["embedding"])
            query_embedding_tensor = torch.tensor(query_embedding)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                query_embedding_tensor.unsqueeze(0),
                chunk_embedding.unsqueeze(0)
            ).item()
            
            similarities.append((i, similarity))
            
        # Sort by similarity and get top_k
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Return the relevant chunks
        relevant_chunks = []
        for idx, similarity in sorted_similarities:
            chunk = dataset[idx]
            chunk_with_score = {**chunk, "similarity": similarity}
            relevant_chunks.append(chunk_with_score)
            
        return relevant_chunks
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer to a question based on the provided context chunks.
        
        Args:
            question: The question to answer
            context_chunks: List of context chunks to use for answering
            
        Returns:
            The generated answer
        """
        # Extract text from chunks
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        # Create prompt for generation
        prompt = f"""Answer the following question based on the context provided. 
Use only information from the context.
If you don't know the answer, say 'I don't have enough information.'

Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            # Import required libraries here to avoid circular imports
            from transformers import pipeline
            
            # Initialize the question answering pipeline with the model
            qa_pipeline = pipeline(
                "text-generation",
                model=self.generation_model_name,
                token=self.api_key,
                max_length=512
            )
            
            # Generate answer
            result = qa_pipeline(prompt, max_length=200, do_sample=False)
            
            # Extract the generated answer
            answer = result[0]['generated_text']
            
            # Clean up the answer (remove the prompt and get just the answer)
            answer = answer.split("Answer:")[-1].strip()
            
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "I was unable to generate an answer due to a technical issue."
