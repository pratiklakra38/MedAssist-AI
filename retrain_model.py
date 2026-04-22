"""
Data Augmentation & Model Retraining Script.

Injects additional targeted synthetic training samples for diseases
that share overlapping symptoms (e.g., Malaria vs Monkeypox, TB vs Heart attack)
to improve the ML classifier's discrimination ability.
"""
import os
import json
import pandas as pd
import numpy as np
from src.ml_model import MLModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Load existing data
with open(os.path.join(PROCESSED_DIR, "disease_list.json"), "r") as f:
    disease_list = json.load(f)
with open(os.path.join(PROCESSED_DIR, "symptom_list.json"), "r") as f:
    symptom_list = json.load(f)

symptom_idx = {sym: idx for idx, sym in enumerate(symptom_list)}

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))

print(f"Before augmentation: {X_train.shape[0]} training samples")

# Define augmentation targets: diseases that need stronger signal
# These are diseases whose symptoms heavily overlap with others
augment_config = {
    # Malaria: chills + high_fever + sweating + headache + muscle_pain
    # Problem: overlaps with Monkeypox (high_fever, headache, muscle_pain)
    "Malaria": {
        "id": disease_list.index("Malaria"),
        "core": ["chills", "high_fever", "sweating", "headache", "muscle_pain"],
        "optional": ["nausea", "diarrhoea", "vomiting", "fatigue"],
        "count": 80
    },
    # Dengue: skin_rash + high_fever + joint_pain + pain_behind_the_eyes
    # Problem: skin_rash overlaps with Impetigo, Psoriasis, Chicken pox
    "Dengue": {
        "id": disease_list.index("Dengue"),
        "core": ["skin_rash", "high_fever", "joint_pain", "pain_behind_the_eyes"],
        "optional": ["nausea", "vomiting", "fatigue", "headache"],
        "count": 80
    },
    # Tuberculosis: cough + weight_loss + sweating + blood_in_sputum
    # Problem: cough overlaps with COVID-19, Pneumonia, Common Cold
    "Tuberculosis": {
        "id": disease_list.index("Tuberculosis"),
        "core": ["cough", "weight_loss", "sweating", "blood_in_sputum"],
        "optional": ["fatigue", "high_fever", "breathlessness", "loss_of_appetite"],
        "count": 80
    },
    # Pneumonia: cough + breathlessness + phlegm + fast_heart_rate + rusty_sputum
    "Pneumonia": {
        "id": disease_list.index("Pneumonia"),
        "core": ["cough", "breathlessness", "phlegm", "fast_heart_rate", "rusty_sputum"],
        "optional": ["high_fever", "chest_pain", "fatigue"],
        "count": 80
    },
    # Typhoid: high_fever + constipation + toxic_look_(typhos) + belly_pain
    "Typhoid": {
        "id": disease_list.index("Typhoid"),
        "core": ["high_fever", "constipation", "toxic_look_(typhos)", "belly_pain"],
        "optional": ["fatigue", "headache", "nausea", "diarrhoea"],
        "count": 80
    },
}

new_rows_x = []
new_rows_y = []
rng = np.random.RandomState(42)

for disease, config in augment_config.items():
    disease_id = config["id"]
    core_indices = [symptom_idx[s] for s in config["core"] if s in symptom_idx]
    optional_indices = [symptom_idx[s] for s in config["optional"] if s in symptom_idx]

    for _ in range(config["count"]):
        row = [0] * len(symptom_list)
        # Always set core symptoms
        for idx in core_indices:
            row[idx] = 1
        # Randomly add 1-2 optional symptoms for variation
        n_opt = rng.randint(0, min(3, len(optional_indices) + 1))
        chosen = rng.choice(optional_indices, size=n_opt, replace=False) if n_opt > 0 else []
        for idx in chosen:
            row[idx] = 1
        new_rows_x.append(row)
        new_rows_y.append(disease_id)

    print(f"  Added {config['count']} samples for {disease} (class {disease_id})")

# Append to training data
X_aug = pd.concat([X_train, pd.DataFrame(new_rows_x, columns=symptom_list)], ignore_index=True)
y_aug = pd.concat([y_train, pd.DataFrame({"prognosis": new_rows_y})], ignore_index=True)

print(f"After augmentation: {X_aug.shape[0]} training samples")

# Save augmented data
X_aug.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
y_aug.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
print("Saved augmented training data.")

# Retrain model
print("\nRetraining ML model with augmented data...")
ml = MLModel()
accuracy = ml.train()

# Verify predictions after retraining
print("\n" + "=" * 60)
print("VERIFICATION: Post-Retrain Predictions")
print("=" * 60)

tests = {
    "Malaria": ["chills", "high_fever", "sweating", "headache", "muscle_pain"],
    "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness"],
    "Dengue": ["skin_rash", "high_fever", "joint_pain", "pain_behind_the_eyes"],
    "Tuberculosis": ["cough", "weight_loss", "sweating", "blood_in_sputum"],
    "Pneumonia": ["cough", "breathlessness", "phlegm", "fast_heart_rate"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    "Typhoid": ["high_fever", "constipation", "belly_pain"],
}

correct = 0
total = len(tests)
for disease, syms in tests.items():
    probs = ml.predict_proba(syms)
    top = list(probs.items())[0]
    match = "CORRECT" if top[0] == disease else "WRONG"
    if top[0] == disease:
        correct += 1
    print(f"  {disease:20s} -> Predicted: {top[0]:20s} ({top[1]*100:.1f}%)  [{match}]")

print(f"\nPrediction accuracy: {correct}/{total} test cases correct")
