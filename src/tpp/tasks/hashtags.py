from transformers import pipeline, logging  
import pandas as pd, re

logging.set_verbosity_error()    

class HashtagGenerator:
    def __init__(self):
        logging.set_verbosity_error()
        self.hashtag_model = pipeline("text-generation", model="gpt2")
        
    
    def generate_hashtags(self, tweet):
        generated = self.hashtag_model(tweet + " #", max_new_tokens=10, num_return_sequences=1)
        return generated[0]['generated_text']
    
    def preprocess(self, df):
        df['content'] = df['content'].astype(str)
        return df
    
    def predict_hashtags(self, df):
        df = self.preprocess(df)
        df['Hashtags'] = df['content'].apply(self.generate_hashtags)
        df['hashtags_final'] = df['Hashtags'].str.findall(r'(#\w+)')
        return df
	
    def predict(self, df: pd.DataFrame):
        """Alias so MTMLModel can call .predict(df)."""
        return self.predict_hashtags(df)

