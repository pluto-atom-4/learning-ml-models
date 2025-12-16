import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    return text.lower().strip()

def preprocess_data(input_path, output_dir):
    df = pd.read_csv(input_path)
    df["clean_text"] = df["text"].apply(clean_text)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
