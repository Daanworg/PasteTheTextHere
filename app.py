import streamlit as st
from text_processor import TextProcessor
from model_handler import ModelHandler
from data_manager import DataManager
from rag_extension import RAGSystem
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize components with proper caching
@st.cache_resource
def init_components():
    try:
        text_processor = TextProcessor()
        model_handler = ModelHandler()
        data_manager = DataManager()
        rag_system = RAGSystem()
        return text_processor, model_handler, data_manager, rag_system
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        logging.error(f"Initialization error: {e}")
        raise

try:
    # Initialize components
    text_processor, model_handler, data_manager, rag_system = init_components()

    # Set up the page
    st.set_page_config(
        page_title="Text Processing App",
        page_icon="📝",
        layout="wide"
    )

    # Main title with description
    st.title("📝 Advanced Text Processing and Analysis")
    st.markdown("""
    This application provides powerful text processing capabilities including:
    - Text cleaning and preprocessing
    - Classification with custom labels
    - Summarization
    - RAG-based question answering
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Processing Options")

        # Text processing options
        with st.expander("Text Processing", expanded=True):
            remove_stopwords = st.checkbox("Remove Stop Words", value=True)
            lemmatize = st.checkbox("Lemmatize Text", value=True)

        # Classification options
        with st.expander("Classification Labels", expanded=True):
            custom_labels = st.text_area(
                "Enter custom labels (one per line)",
                value="business\ntechnology\nhealth\nentertainment",
                height=100
            )
            custom_labels = [label.strip() for label in custom_labels.split("\n") if label.strip()]

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Text Processing", "Search & History", "RAG Q&A"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Text")
            input_text = st.text_area("Paste your text here:", height=200)

            if st.button("Process Text"):
                if input_text:
                    try:
                        with st.spinner("Processing text..."):
                            # Process text
                            processed_text = text_processor.clean_text(
                                input_text,
                                remove_stopwords=remove_stopwords,
                                lemmatize=lemmatize
                            )

                            # Get text statistics
                            stats = text_processor.get_text_stats(input_text)

                            # Generate summary
                            summary = model_handler.summarize_text(input_text)

                            # Classify text
                            classification_results = model_handler.classify_text(
                                processed_text,
                                labels=custom_labels
                            )

                            # Save results
                            text_id = data_manager.save_processed_text(
                                input_text,
                                processed_text,
                                classification_results,
                                summary
                            )

                            st.success("Text processed successfully!")

                        with col2:
                            st.subheader("Processing Results")

                            with st.expander("Processed Text", expanded=True):
                                st.write(processed_text)

                            with st.expander("Summary", expanded=True):
                                st.write(summary)

                            with st.expander("Text Statistics", expanded=True):
                                for key, value in stats.items():
                                    st.metric(
                                        key.replace('_', ' ').title(),
                                        f"{value:.2f}" if isinstance(value, float) else value
                                    )

                            with st.expander("Classification Results", expanded=True):
                                for label, score in zip(
                                    classification_results['labels'],
                                    classification_results['scores']
                                ):
                                    st.progress(score)
                                    st.write(f"{label}: {score:.2%}")

                    except Exception as e:
                        st.error(f"Error processing text: {str(e)}")
                        logging.error(f"Processing error: {e}")
                else:
                    st.warning("Please enter some text to process.")

    with tab2:
        st.subheader("Search Previous Entries")
        search_col1, search_col2, search_col3 = st.columns(3)

        with search_col1:
            search_query = st.text_input("Search by content:")
        with search_col2:
            search_label = st.selectbox("Filter by label:", [""] + custom_labels)
        with search_col3:
            min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.5)

        if search_query or search_label:
            try:
                results = data_manager.search_texts(search_query, search_label, min_confidence)
                if results:
                    st.write(f"Found {len(results)} matching entries:")
                    for result in results:
                        with st.expander(f"Entry {result['id']} - {result['timestamp']}"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Original Text:**")
                                st.write(result['original_text'])
                            with cols[1]:
                                st.write("**Processed Text:**")
                                st.write(result['processed_text'])
                                st.write("**Classifications:**")
                                labels = eval(result['classification'])
                                scores = eval(result['confidence_score'])
                                for label, score in zip(labels, scores):
                                    st.write(f"- {label}: {score:.2%}")
                else:
                    st.info("No matching entries found.")
            except Exception as e:
                st.error(f"Error searching entries: {str(e)}")
                logging.error(f"Search error: {e}")

    with tab3:
        st.subheader("Question Answering with RAG")

        # Context input
        context = st.text_area("Enter or paste context document:", height=150)
        question = st.text_input("Ask a question about the context:")

        if st.button("Get Answer") and context and question:
            try:
                with st.spinner("Processing..."):
                    # Process document with RAG
                    processed_doc = rag_system.process_document(context)

                    # Create in-memory dataset
                    from datasets import Dataset
                    dataset = Dataset.from_list([
                        {"chunk_id": i, "text": chunk["text"], "embedding": chunk["embedding"]}
                        for i, chunk in enumerate(processed_doc["chunks"])
                    ])

                    # Retrieve relevant chunks and generate answer
                    relevant_chunks = rag_system.retrieve_relevant_chunks(question, dataset)
                    answer = rag_system.answer_question(question, relevant_chunks)

                    # Display results
                    st.write("**Answer:**")
                    st.write(answer)

                    with st.expander("View relevant context"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"**Chunk {i+1}** (Similarity: {chunk['similarity']:.2f})")
                            st.write(chunk['text'])
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                logging.error(f"RAG error: {e}")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    logging.error(f"Application error: {e}")