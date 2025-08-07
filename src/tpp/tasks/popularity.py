from transformers import DistilBertTokenizer, DistilBertModel, pipeline, logging  
import pandas as pd, numpy as np, re, torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle  

logging.set_verbosity_error()    

class PopularityPredictor:
    def __init__(self):
        logging.set_verbosity_error()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.t_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = LinearRegression()
        self.scaler = MinMaxScaler(feature_range=(0, 100))
    
    def preprocess(self, df):
        df['number_of_shares'] = df['number_of_shares'].astype(float)
        df['number_of_likes'] = df['number_of_likes'].astype(float)
        df['content'] = df['content'].astype(str)
        df['hashtags_final'] = df['hashtags_final'].astype(str)
        df['content_length'] = df['content'].apply(len)
        df['hashtags_count'] = df['hashtags_final'].apply(lambda tweet: len(re.findall(r'#\w+', tweet)))
        return df
    
    def get_text_embeddings(self, tweets):
        embeddings = []
        for tweet in tweets:
            inputs = self.tokenizer(tweet, return_tensors='pt', padding=True, truncation=True, max_new_tokens=10)
            with torch.no_grad():
                output = self.t_model(**inputs)
                embeddings.append(output.last_hidden_state[:, 0, :].numpy())
        return np.array(embeddings).reshape(len(embeddings), -1)
    
    def train_model(self, df):
        df = self.preprocess(df)
        X = df[['content', 'number_of_shares', 'number_of_likes', 'content_length', 'hashtags_count']]
        tweet_embeddings = self.get_text_embeddings(X['content'])
        X_combined = np.hstack((tweet_embeddings, X[['number_of_shares', 'number_of_likes', 'content_length', 'hashtags_count']].values))
        y_simulated = (X['number_of_likes'] + X['number_of_shares']) / 2
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_simulated, test_size=0.2, random_state=42)
        self.regressor.fit(X_train, y_train)
        y_pred = self.regressor.predict(X_test)
        return self.scaler.fit_transform(y_pred.reshape(-1, 1))
    
    def inference(self, df):
        df = self.preprocess(df)
        X = df[['content', 'number_of_shares', 'number_of_likes', 'content_length', 'hashtags_count']]
        tweet_embeddings = self.get_text_embeddings(X['content'])
        X_combined = np.hstack((tweet_embeddings, X[['number_of_shares', 'number_of_likes', 'content_length', 'hashtags_count']].values))
        popularity_predictions = self.regressor.predict(X_combined)
        df['popularity'] = self.scaler.transform(popularity_predictions.reshape(-1, 1))

        return df

    def export_model(self, filename='popularity_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.regressor, f)
    
    def load_model(self, filename='popularity_model.pkl'):
        with open(filename, 'rb') as f:
            self.regressor = pickle.load(f)

