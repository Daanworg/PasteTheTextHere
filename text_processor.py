import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging
import os

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        """Download required NLTK data with proper error handling"""
        # Create nltk_data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)

        required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for package in required_packages:
            try:
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)
                logging.info(f"Successfully downloaded NLTK package: {package}")
            except Exception as e:
                logging.error(f"Failed to download NLTK package {package}: {e}")
                raise RuntimeError(f"Failed to initialize NLTK: {str(e)}")

    def clean_text(self, text, remove_stopwords=True, lemmatize=True):
        """Clean and preprocess text"""
        if not text:
            return ""

        try:
            # Convert to lowercase
            text = text.lower()

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize using word_tokenize directly
            try:
                tokens = word_tokenize(text)
            except LookupError:
                # Fallback to basic splitting if NLTK tokenization fails
                tokens = text.split()
                logging.warning("Falling back to basic tokenization")

            # Remove stopwords if requested
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]

            # Lemmatize if requested
            if lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

            # Join tokens back into text
            cleaned_text = ' '.join(tokens)
            return cleaned_text
        except Exception as e:
            logging.error(f"Text cleaning error: {e}")
            raise RuntimeError(f"Failed to clean text: {str(e)}")

    def get_text_stats(self, text):
        """Get basic statistics about the text"""
        if not text:
            return {}

        try:
            # Use basic splitting as fallback if tokenization fails
            try:
                words = word_tokenize(text)
                sentences = sent_tokenize(text)
            except LookupError:
                words = text.split()
                sentences = text.split('.')
                logging.warning("Using basic text statistics due to NLTK tokenization failure")

            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'unique_words': len(set(words))
            }
        except Exception as e:
            logging.error(f"Error calculating text stats: {e}")
            raise RuntimeError(f"Failed to calculate text statistics: {str(e)}")