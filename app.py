import streamlit as st
from text_processor import TextProcessor
from model_handler import ModelHandler
from data_manager import DataManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize components
@st.cache_resource
def init_components():
    try:
        text_processor = TextProcessor()
        model_handler = ModelHandler()
        data_manager = DataManager()
        return text_processor, model_handler, data_manager
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        raise

try:
    text_processor, model_handler, data_manager = init_components()

    # Set up the page
    st.set_page_config(
        page_title="Text Processing App",
        page_icon="📝",
        layout="wide"
    )

    # Main title
    st.title("📝 Text Processing and Classification App")

    # Sidebar
    st.sidebar.header("Processing Options")
    remove_stopwords = st.sidebar.checkbox("Remove Stop Words", value=True)
    lemmatize = st.sidebar.checkbox("Lemmatize Text", value=True)

    # Custom labels input
    st.sidebar.header("Classification Labels")
    custom_labels = st.sidebar.text_area(
        "Enter custom labels (one per line)",
        value="business\ntechnology\nhealth\nentertainment",
        height=100
    )
    custom_labels = [label.strip() for label in custom_labels.split("\n") if label.strip()]

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Text")
        input_text = st.text_area("Paste your text here:", height=200)

        if st.button("Process Text"):
            if input_text:
                try:
                    with st.spinner("Processing text..."):
                        # Clean and process text
                        processed_text = text_processor.clean_text(
                            input_text,
                            remove_stopwords=remove_stopwords,
                            lemmatize=lemmatize
                        )

                        # Get text statistics
                        stats = text_processor.get_text_stats(input_text)

                        # Classify text
                        classification_results = model_handler.classify_text(
                            processed_text,
                            labels=custom_labels
                        )

                        # Save results
                        text_id = data_manager.save_processed_text(
                            input_text,
                            processed_text,
                            classification_results
                        )

                        # Display results
                        st.success("Text processed successfully!")

                        with col2:
                            st.subheader("Processing Results")
                            st.write("**Processed Text:**")
                            st.write(processed_text)

                            st.write("**Text Statistics:**")
                            for key, value in stats.items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value:.2f}")

                            st.write("**Classification Results:**")
                            for label, score in zip(
                                classification_results['labels'],
                                classification_results['scores']
                            ):
                                st.write(f"- {label}: {score:.2%}")

                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
                    logging.error(f"Processing error: {e}")
            else:
                st.warning("Please enter some text to process.")

    # Search and Filter Section
    st.header("Search Previous Entries")
    search_col1, search_col2 = st.columns(2)

    with search_col1:
        search_query = st.text_input("Search by content:")

    with search_col2:
        search_label = st.selectbox("Filter by label:", [""] + custom_labels)

    if search_query or search_label:
        try:
            results = data_manager.search_texts(search_query, search_label)
            if results:
                st.write(f"Found {len(results)} matching entries:")
                for result in results:
                    with st.expander(f"Entry {result['id']} - {result['timestamp']}"):
                        st.write("**Original Text:**")
                        st.write(result['original_text'])
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

except Exception as e:
    st.error(f"Application error: {str(e)}")
    logging.error(f"Application error: {e}")