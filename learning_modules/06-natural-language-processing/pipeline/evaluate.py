import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(test_path, model_dir, output_dir):
    df = pd.read_csv(test_path)

    vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")
    model = joblib.load(f"{model_dir}/sentiment_model.pkl")

    X_test = vectorizer.transform(df["clean_text"])
    y_test = df["label"]

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(report, f, indent=4)
