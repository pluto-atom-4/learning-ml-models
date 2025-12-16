import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(train_path, model_dir):
    df = pd.read_csv(train_path)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    joblib.dump(vectorizer, f"{model_dir}/vectorizer.pkl")
    joblib.dump(model, f"{model_dir}/sentiment_model.pkl")
