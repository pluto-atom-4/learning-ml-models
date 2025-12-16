import os
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, RAW_DATA, TRAIN_DATA, TEST_DATA
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

def run_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    preprocess_data(
        input_path=RAW_DATA,
        output_dir=DATA_DIR
    )

    train_model(
        train_path=TRAIN_DATA,
        model_dir=MODEL_DIR
    )

    evaluate_model(
        test_path=TEST_DATA,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )

if __name__ == "__main__":
    run_pipeline()
