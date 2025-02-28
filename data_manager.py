import pandas as pd
import json
from datetime import datetime
import os

class DataManager:
    def __init__(self):
        self.data_file = "processed_texts.csv"
        self.initialize_storage()

    def initialize_storage(self):
        """Initialize the storage file if it doesn't exist"""
        if not os.path.exists(self.data_file):
            df = pd.DataFrame(columns=[
                'id', 'original_text', 'processed_text', 
                'classification', 'confidence_score',
                'timestamp', 'labels'
            ])
            df.to_csv(self.data_file, index=False)

    def save_processed_text(self, original_text, processed_text, classification_results):
        """Save processed text and its metadata"""
        try:
            df = pd.read_csv(self.data_file)
            
            new_entry = {
                'id': len(df) + 1,
                'original_text': original_text,
                'processed_text': processed_text,
                'classification': json.dumps(classification_results.get('labels', [])),
                'confidence_score': json.dumps(classification_results.get('scores', [])),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'labels': json.dumps(classification_results)
            }
            
            df = df.append(new_entry, ignore_index=True)
            df.to_csv(self.data_file, index=False)
            return new_entry['id']
        except Exception as e:
            raise Exception(f"Error saving data: {e}")

    def search_texts(self, query=None, label=None):
        """Search stored texts by query or label"""
        try:
            df = pd.read_csv(self.data_file)
            
            if query:
                df = df[df['processed_text'].str.contains(query, case=False, na=False)]
            
            if label:
                df = df[df['labels'].apply(lambda x: label in str(x))]
            
            return df.to_dict('records')
        except Exception as e:
            raise Exception(f"Error searching data: {e}")
