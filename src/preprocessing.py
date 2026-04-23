import pandas as pd
import json
import os

base = os.path.join(os.path.dirname(__file__), "..")
raw_path = os.path.join(base, "data", "raw")
out_path = os.path.join(base, "data", "processed")

def clean_and_save():
    os.makedirs(out_path, exist_ok=True)

    train = pd.read_csv(os.path.join(raw_path, "Training.csv"))
    test = pd.read_csv(os.path.join(raw_path, "Testing.csv"))

    for df in [train, test]:
        junk = [c for c in df.columns if "Unnamed" in c or c.endswith(".1")]
        df.drop(columns=junk, inplace=True)

    train.columns = [c.strip().lower().replace(" ", "_") for c in train.columns]
    test.columns = [c.strip().lower().replace(" ", "_") for c in test.columns]

    symptoms = sorted([c for c in train.columns if c != "prognosis"])
    diseases = sorted(train["prognosis"].unique().tolist())
    label_map = {name: i for i, name in enumerate(diseases)}

    train[symptoms].to_csv(os.path.join(out_path, "X_train.csv"), index=False)
    test[symptoms].to_csv(os.path.join(out_path, "X_test.csv"), index=False)
    train["prognosis"].map(label_map).to_csv(os.path.join(out_path, "y_train.csv"), index=False)
    test["prognosis"].map(label_map).to_csv(os.path.join(out_path, "y_test.csv"), index=False)

    with open(os.path.join(out_path, "symptom_list.json"), "w") as f:
        json.dump(symptoms, f, indent=2)
    with open(os.path.join(out_path, "disease_list.json"), "w") as f:
        json.dump(diseases, f, indent=2)

    print(f"{len(symptoms)} symptoms, {len(diseases)} diseases")
    print(f"Saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    clean_and_save()
