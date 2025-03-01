from transformers import pipeline
import os
import logging
import torch

class ModelHandler:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")

        logging.info("Initializing ModelHandler with HuggingFace API key")
        self.classifier = None
        self.zero_shot_classifier = None
        self.summarizer = None

        # Initialize models with better error handling
        try:
            self.initialize_models()
        except Exception as e:
            logging.error(f"Failed to initialize models: {e}")
            raise

    def initialize_models(self):
        """Initialize the classification models with detailed error logging"""
        try:
            logging.info("Starting model initialization...")

            # Verify PyTorch installation
            logging.info(f"PyTorch version: {torch.__version__}")
            logging.info(f"CUDA available: {torch.cuda.is_available()}")

            # Text classification pipeline
            logging.info("Initializing text classification pipeline...")
            self.classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                token=self.api_key,
                device=-1  # Force CPU usage
            )
            logging.info("Text classification pipeline initialized successfully")

            # Zero-shot classification pipeline
            logging.info("Initializing zero-shot classification pipeline...")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                token=self.api_key,
                device=-1  # Force CPU usage
            )
            logging.info("Zero-shot classification pipeline initialized successfully")

            # Summarization pipeline
            logging.info("Initializing summarization pipeline...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                token=self.api_key,
                device=-1  # Force CPU usage
            )
            logging.info("Summarization pipeline initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            logging.error("Model initialization failed. Please check your HuggingFace API key and network connection.")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def classify_text(self, text, labels=None):
        """Classify text using pre-defined or custom labels"""
        if not text:
            return {}

        try:
            logging.info("Starting text classification...")
            if labels:
                # Use zero-shot classification with custom labels
                if not self.zero_shot_classifier:
                    raise RuntimeError("Zero-shot classifier not initialized")

                logging.info(f"Performing zero-shot classification with labels: {labels}")
                result = self.zero_shot_classifier(
                    text,
                    candidate_labels=labels,
                    multi_label=True
                )
                return {
                    'labels': result['labels'],
                    'scores': result['scores']
                }
            else:
                # Use standard classification
                if not self.classifier:
                    raise RuntimeError("Text classifier not initialized")

                logging.info("Performing standard text classification")
                result = self.classifier(text)
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score']
                }
        except Exception as e:
            logging.error(f"Classification error: {str(e)}")
            raise RuntimeError(f"Failed to classify text: {str(e)}")

    def summarize_text(self, text, max_length=150, min_length=30):
        """Generate a summary of the input text"""
        if not text:
            return ""

        try:
            logging.info("Starting text summarization...")
            if not self.summarizer:
                raise RuntimeError("Summarizer not initialized")

            # Calculate appropriate max_length based on input text length
            input_length = len(text.split())
            max_length = min(max_length, input_length // 2)  # Summary should not be longer than half the input
            min_length = min(min_length, max_length - 1)  # Ensure min_length is less than max_length

            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            return result[0]['summary_text']
        except Exception as e:
            logging.error(f"Summarization error: {str(e)}")
            raise RuntimeError(f"Failed to summarize text: {str(e)}")