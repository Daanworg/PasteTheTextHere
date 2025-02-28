from transformers import pipeline
import os

class ModelHandler:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.initialize_models()

    def initialize_models(self):
        """Initialize the classification models"""
        try:
            # Text classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                token=self.api_key
            )
            
            # Zero-shot classification pipeline
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                token=self.api_key
            )
        except Exception as e:
            raise Exception(f"Error initializing models: {e}")

    def classify_text(self, text, labels=None):
        """Classify text using pre-defined or custom labels"""
        if not text:
            return {}

        try:
            if labels:
                # Use zero-shot classification with custom labels
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
                result = self.classifier(text)
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score']
                }
        except Exception as e:
            raise Exception(f"Classification error: {e}")
