import os
import json
import pandas as pd
import numpy as np
from src.ml_model import MLModel

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "data", "processed")
    
    # 1. Load lists
    with open(os.path.join(processed_dir, "disease_list.json"), "r") as f:
        disease_list = json.load(f)
    with open(os.path.join(processed_dir, "symptom_list.json"), "r") as f:
        symptom_list = json.load(f)
        
    symptom_idx = {sym: idx for idx, sym in enumerate(symptom_list)}
    
    # 2. Add New Diseases
    new_diseases = ["COVID-19", "Monkeypox"]
    added = False
    for nd in new_diseases:
        if nd not in disease_list:
            disease_list.append(nd)
            added = True
            
    if added:
        with open(os.path.join(processed_dir, "disease_list.json"), "w") as f:
            json.dump(disease_list, f, indent=4)
            
    # 3. Load Datasets
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))
    
    # 4. Generate Synthetic Data
    synthetic_config = {
        "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness"],
        "Monkeypox": ["high_fever", "skin_rash", "swelled_lymph_nodes", "headache", "muscle_pain"]
    }
    
    new_rows_x_train = []
    new_rows_y_train = []
    new_rows_x_test = []
    new_rows_y_test = []
    
    for disease, required_symptoms in synthetic_config.items():
        disease_id = disease_list.index(disease)
        req_indices = [symptom_idx[sym] for sym in required_symptoms if sym in symptom_idx]
        
        # 100 for train
        for _ in range(100):
            row = [0] * len(symptom_list)
            for idx in req_indices:
                row[idx] = 1
            if np.random.rand() > 0.5:
                row[np.random.randint(0, len(symptom_list))] = 1
            new_rows_x_train.append(row)
            new_rows_y_train.append(disease_id)
            
        # 20 for test
        for _ in range(20):
            row = [0] * len(symptom_list)
            for idx in req_indices:
                row[idx] = 1
            if np.random.rand() > 0.5:
                row[np.random.randint(0, len(symptom_list))] = 1
            new_rows_x_test.append(row)
            new_rows_y_test.append(disease_id)
            
    # Append
    X_train = pd.concat([X_train, pd.DataFrame(new_rows_x_train, columns=symptom_list)], ignore_index=True)
    y_train = pd.concat([y_train, pd.DataFrame({"prognosis": new_rows_y_train})], ignore_index=True)
    
    X_test = pd.concat([X_test, pd.DataFrame(new_rows_x_test, columns=symptom_list)], ignore_index=True)
    y_test = pd.concat([y_test, pd.DataFrame({"prognosis": new_rows_y_test})], ignore_index=True)
    
    # Save back
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    print(f"Injected {len(new_rows_x_train)} train rows and {len(new_rows_x_test)} test rows. Now retraining ML model...")
    
    # 5. Retrain ML
    ml = MLModel()
    ml.train()

if __name__ == "__main__":
    main()
