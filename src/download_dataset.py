"""
Dataset Acquisition Script for MedAssist AI
Downloads the Kaggle Disease-Symptom Prediction dataset using kagglehub.

Dataset: ~40 diseases, 132 binary symptom columns, 1 target column (prognosis)
Source: Kaggle - "Disease Prediction Using Machine Learning"
"""

import os
import shutil
import kagglehub


def download_dataset():
    """Download the disease-symptom dataset from Kaggle and place it in data/raw/."""

    # Get the project root (parent of the directory this script is in)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    print("(If prompted, set up Kaggle credentials via KAGGLE_USERNAME and KAGGLE_KEY env vars)")
    print()

    try:
        # Download dataset using kagglehub
        path = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
        print(f"Dataset downloaded to: {path}")

        # Copy all CSV files to data/raw/
        for file in os.listdir(path):
            if file.endswith(".csv"):
                src = os.path.join(path, file)
                dst = os.path.join(raw_data_dir, file)
                shutil.copy2(src, dst)
                print(f"  Copied: {file} -> data/raw/{file}")

        print("\nDataset acquisition complete!")
        print(f"Files saved to: {raw_data_dir}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print()
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Go to: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning")
        print("2. Download the dataset (ZIP)")
        print("3. Extract CSV files into: data/raw/")
        print()
        raise


if __name__ == "__main__":
    download_dataset()
