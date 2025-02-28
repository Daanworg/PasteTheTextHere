import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text, remove_stopwords=True, lemmatize=True):
        """Clean and preprocess text"""
        if not text:
            return ""
        
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

    def get_text_stats(self, text):
        """Get basic statistics about the text"""
        if not text:
            return {}
        
        words = word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words),
            'unique_words': len(set(words))
        }
