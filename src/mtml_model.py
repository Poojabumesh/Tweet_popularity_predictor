import argparse
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel, logging, pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
import pickle


class Task1EmotionClassification:
    def __init__(self):
        logging.set_verbosity_error()
        self.emotion_classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')
        
    
    def classify_tweet(self, tweet):
        return self.emotion_classifier(tweet)[0]['label']
    
    def preprocess(self, df):
        df['content'] = df['content'].astype(str)
        return df
    
    def predict_emotion(self, df):
        df = self.preprocess(df)
        df['Emotion'] = df['content'].apply(self.classify_tweet).apply(pd.Series)
        return df


class Task2HashtagGeneration:
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


class Task3PopularityPrediction:
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


class MTMLModel:
    def __init__(self):
        self.task1 = Task1EmotionClassification()
        self.task2 = Task2HashtagGeneration()
        self.task3 = Task3PopularityPrediction()

    def predict_emotion(self, df):
        return self.task1.predict_emotion(df)

    def predict_hashtags(self, df):
        return self.task2.predict_hashtags(df)

    def predict_popularity(self, df):
        # Ensure predict_hashtags is called before popularity
        df = self.predict_hashtags(df)
        self.task3.train_model(df)
        self.task3.export_model()
        predictions = self.task3.inference(df)
        return predictions

    def load_popularity_model(self):
        self.task3.load_model()


def main(args):
    if args.file:
        df = pd.read_csv(args.file, nrows=10000)
    else:
        data = {'content': [args.content], 'number_of_shares': [args.number_of_shares], 'number_of_likes': [args.number_of_likes]}
        df = pd.DataFrame(data)

    mtml_model = MTMLModel()

    if args.task == 'emotion':
        result = mtml_model.predict_emotion(df)
    elif args.task == 'hashtags':
        result = mtml_model.predict_hashtags(df)
    elif args.task == 'popularity':
        result = mtml_model.predict_popularity(df)
    elif args.task == 'all':
        df = mtml_model.predict_emotion(df)
        df = mtml_model.predict_hashtags(df)
        df = mtml_model.predict_popularity(df)
        result = df
    else:
        raise ValueError("Task must be one of ['emotion', 'hashtags', 'popularity', 'all]")
    
    # Save the result to a CSV file with headers
    result.to_csv('final_result.csv', index=False, header=True)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTML Model for multi-task predictions')
    parser.add_argument('--file', type=str, help='Path to the input CSV file')
    parser.add_argument('--content', type=str, help='Tweet content for prediction')
    parser.add_argument('--number_of_shares', type=int, help='Number of shares for the tweet')
    parser.add_argument('--number_of_likes', type=int, help='Number of likes for the tweet')
    parser.add_argument('--task', type=str, choices=['emotion', 'hashtags', 'popularity', 'all'], required=True, help='Task to predict')

    args = parser.parse_args()
    main(args)
