"""Diagnostic: test ML model accuracy for Malaria and other critical diseases."""
import json, os
from src.ml_model import MLModel

ml = MLModel()

# Check class mapping
print("=" * 60)
print("CLASS MAPPING CHECK")
print("=" * 60)
print(f"model.classes_ type: {type(ml.model.classes_)}")
print(f"model.classes_ length: {len(ml.model.classes_)}")
print(f"disease_list length: {len(ml.disease_list)}")
print(f"\nFirst 5 model classes: {ml.model.classes_[:5]}")
print(f"Last 5 model classes: {ml.model.classes_[-5:]}")
print(f"\nDisease list sample:")
for i, d in enumerate(ml.disease_list):
    print(f"  [{i}] {d}")

# Test specific disease symptom combos through raw ML
print("\n" + "=" * 60)
print("RAW ML PREDICTIONS (predict_proba)")
print("=" * 60)

tests = {
    "Malaria": ["chills", "high_fever", "sweating", "headache", "muscle_pain"],
    "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness"],
    "Dengue": ["skin_rash", "high_fever", "joint_pain", "pain_behind_the_eyes"],
    "Tuberculosis": ["cough", "weight_loss", "sweating", "blood_in_sputum"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
}

for disease, syms in tests.items():
    probs = ml.predict_proba(syms)
    top3 = list(probs.items())[:3]
    print(f"\nExpected: {disease}")
    print(f"  Symptoms: {syms}")
    print(f"  Top 3 ML predictions:")
    for rank, (d, p) in enumerate(top3, 1):
        marker = " <-- CORRECT" if d == disease else ""
        print(f"    {rank}. {d}: {p*100:.1f}%{marker}")
    if top3[0][0] != disease:
        print(f"  *** MISMATCH: ML says '{top3[0][0]}' but expected '{disease}' ***")
