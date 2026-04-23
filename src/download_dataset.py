import kagglehub
import shutil
import os

save_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def download():
    os.makedirs(save_path, exist_ok=True)
    downloaded = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
    for file in os.listdir(downloaded):
        if file.endswith(".csv"):
            shutil.copy2(os.path.join(downloaded, file), save_path)
    print(f"Done! Files saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    download()
