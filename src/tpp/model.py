from .tasks.emotion    import EmotionClassifier
from .tasks.hashtags   import HashtagGenerator
from .tasks.popularity import PopularityPredictor
import pandas as pd


class MTMLModel:
    """Facade that stitches the three task modules together."""

    def __init__(self):
        self.emotion    = EmotionClassifier()
        self.hashtags   = HashtagGenerator()
        self.popularity = PopularityPredictor()

    # ---------- thin wrappers ----------
    def predict_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.emotion.predict(df)

    def predict_hashtags(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.hashtags.predict(df)

    def predict_popularity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Ensure hashtag features exist
        2. Train regressor + scaler (in-memory)
        3. Persist the regressor for future reuse
        4. Run inference and return enriched DataFrame
        """
        df = self.predict_hashtags(df)            # step 1
        self.popularity.train_model(df)           # step 2
        self.popularity.export_model()            # step 3
        return self.popularity.inference(df)      # step 4

    def load_popularity_model(self, filename: str = "popularity_model.pkl") -> None:
        """Reload a previously trained regressor + scaler."""
        self.popularity.load_model(filename)
