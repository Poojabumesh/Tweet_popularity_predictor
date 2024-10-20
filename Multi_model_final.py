from transformers import DistilBertTokenizer, DistilBertModel, logging, pipeline

class Task1EmotionClassification:
    def __init__(self):
        logging.set_verbosity_error() #to display only errors
        #initiating the Hugging face emotion classifier
        self.emotion_classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')
        
    
    def classify_tweet(self, tweet):
        return self.emotion_classifier(tweet)[0]['label']
    
    #to avoid the error regarding type
    def preprocess(self, df):
        df['content'] = df['content'].astype(str)
        return df
    
    def predict_emotion(self, df):
        df = self.preprocess(df)
        df['Emotion'] = df['content'].apply(self.classify_tweet).apply(pd.Series)
        return df
    
    class Task2HashtagPrediction:
        