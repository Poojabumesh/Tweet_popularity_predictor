from transformers import pipeline, logging
import pandas as pd

logging.set_verbosity_error()

class EmotionClassifier:
    """DistilBERT emotion classification on tweet text."""
    
    def __init__(self, model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"):
        self.pipe = pipeline("text-classification", model=model_name)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df["content"] = df["content"].astype(str)
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(df)
        df["Emotion"] = df["content"].apply(lambda t: self.pipe(t)[0]["label"])
        return df
