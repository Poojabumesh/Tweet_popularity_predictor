import argparse, pandas as pd
from .model import MTMLModel

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tweet Popularity Predictor")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="CSV file with tweets")
    g.add_argument("--content", help="Single tweet text")
    p.add_argument("--number_of_shares", type=int, default=0)
    p.add_argument("--number_of_likes",  type=int, default=0)
    p.add_argument("--task", choices=["emotion", "hashtags", "popularity", "all"],
                   required=True)
    return p

def main() -> None:
    args = build_parser().parse_args()
    
    if args.file:
        df = pd.read_csv(args.file, nrows=10_000)
    else:
        df = pd.DataFrame(
            {"content": [args.content],
             "number_of_shares": [args.number_of_shares],
             "number_of_likes":  [args.number_of_likes]}
        )

    model = MTMLModel()
    task = args.task
    if task == "emotion":
        out = model.predict_emotion(df)
    elif task == "hashtags":
        out = model.predict_hashtags(df)
    elif task == "popularity":
        out = model.predict_popularity(df)
    else:   # all
        out = model.predict_popularity(model.predict_hashtags(model.predict_emotion(df)))

    out.to_csv("final_result.csv", index=False)
    print(out)

if __name__ == "__main__":
    main()
