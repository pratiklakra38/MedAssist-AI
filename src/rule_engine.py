import json
import os

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

rules = {
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
    "Dengue": ["skin_rash", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain"],
    "Typhoid": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea"],
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "loss_of_smell"],
    "Pneumonia": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    "Diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite", "polyuria"],
    "Hypertension": ["headache", "chest_pain", "dizziness", "lack_of_concentration", "loss_of_balance"],
    "Jaundice": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
    "Hepatitis B": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellow_urine", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"],
    "Hepatitis C": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "family_history"],
    "Tuberculosis": ["chills", "vomiting", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "weight_loss", "malaise", "phlegm", "chest_pain", "blood_in_sputum", "loss_of_appetite"],
    "Migraine": ["acidity", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability", "visual_disturbances"],
    "Heart Attack": ["vomiting", "breathlessness", "sweating", "chest_pain"],
    "Chicken Pox": ["itching", "skin_rash", "fatigue", "high_fever", "headache", "lethargy", "malaise", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "red_spots_over_body"],
    "Urinary Tract Infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    "Gastroenteritis": ["vomiting", "dehydration", "diarrhoea", "abdominal_pain", "nausea"],
    "Bronchial Asthma": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
    "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
    "Drug Reaction": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting__urination"],
    "Fungal Infection": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic__patches"],
}


class RuleEngine:
    def __init__(self):
        with open(os.path.join(data_path, "symptom_list.json")) as f:
            self.symptoms = json.load(f)

    def evaluate(self, user_symptoms):
        active = {self.symptoms[i] for i, v in enumerate(user_symptoms) if v == 1}

        matches = {}
        for disease, needed in rules.items():
            hit = set(needed) & active
            if not hit:
                continue
            score = round((len(hit) / len(needed)) * 100, 1)
            matches[disease] = {
                "confidence": score,
                "matched_symptoms": sorted(hit),
                "matched_count": len(hit)
            }

        return dict(sorted(matches.items(), key=lambda x: x[1]["confidence"], reverse=True))
