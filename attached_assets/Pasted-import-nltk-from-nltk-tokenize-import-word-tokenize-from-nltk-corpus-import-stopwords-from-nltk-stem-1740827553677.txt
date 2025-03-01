import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        """Download required NLTK data with proper error handling"""
        required_packages = ['punkt', 'stopwords', 'wordnet']
        for package in required_packages:
            try:
                nltk.download(package, quiet=True)
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

            # Tokenize
            tokens = word_tokenize(text)

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
            words = word_tokenize(text)
            sentences = nltk.sent_tokenize(text)

            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'unique_words': len(set(words))
            }
        except Exception as e:
            logging.error(f"Error calculating text stats: {e}")
            raise RuntimeError(f"Failed to calculate text statistics: {str(e)}")