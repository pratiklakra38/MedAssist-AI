"""
MedAssist AI — Rule-Based Inference Engine (Forward Chaining)

Implements a Forward Chaining reasoning strategy:
- Starts from KNOWN FACTS (patient symptoms)
- Progressively matches all applicable IF-THEN rules
- Surfaces multiple candidate diagnoses ranked by confidence

This is data-driven inference (vs. backward chaining which requires
a hypothesis upfront). Forward chaining is appropriate for general-purpose
screening where we don't know the target disease beforehand.

References:
- Shortliffe & Buchanan (1975) - MYCIN Expert System
- WHO ICD-11 (2022) - Clinical diagnostic criteria
- ICMR Protocol (2023) - India-specific clinical guidelines
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RuleMatch:
    """Represents a single rule match result from Forward Chaining."""
    disease: str
    match_strength: float
    matched_symptoms: List[str]

    def to_dict(self):
        return {
            "disease": self.disease,
            "match_strength": self.match_strength,
            "matched_symptoms": self.matched_symptoms
        }

class RuleEngine:
    """
    Deterministic IF-THEN Forward Chaining Inference Engine.

    Forward Chaining Process:
    1. Accept known facts (patient-reported symptoms)
    2. Iterate through all rules in the knowledge base
    3. For each rule, compute overlap ratio between patient facts and required symptoms
    4. If overlap >= threshold, the rule fires and produces a diagnosis candidate
    5. All fired rules are ranked by match strength (confidence weight)

    The rule base is derived from WHO ICD-11 diagnostic criteria and ICMR
    protocols for India-specific disease prevalence.
    """
    def __init__(self, rules_path: str = None):
        self.rules = {}
        
        # Knowledge Base: IF-THEN rules derived from WHO ICD-11 & ICMR guidelines
        # Each disease maps to its required diagnostic symptom markers
        self.default_rules = {
            # --- Kaggle Dataset Diseases (ICD-11 aligned) ---
            "Fungal infection": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches"],
            "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
            "GERD": ["acidity", "ulcers_on_tongue", "vomiting", "chest_pain"],
            "Chronic cholestasis": ["itching", "yellowish_skin", "abdominal_pain", "yellowing_of_eyes"],
            "Drug Reaction": ["itching", "skin_rash", "burning_micturition", "spotting_ urination"],
            "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
            "Bronchial Asthma": ["fatigue", "breathlessness", "family_history", "mucoid_sputum"],
            "Migraine": ["headache", "blurred_and_distorted_vision", "stiff_neck", "visual_disturbances"],
            "Cervical spondylosis": ["weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
            "Jaundice": ["weight_loss", "high_fever", "yellowish_skin", "dark_urine"],
            "Malaria": ["chills", "high_fever", "sweating", "headache", "muscle_pain"],
            "Chicken pox": ["itching", "skin_rash", "mild_fever", "red_spots_over_body"],
            "Dengue": ["skin_rash", "high_fever", "joint_pain", "pain_behind_the_eyes"],
            "Typhoid": ["high_fever", "constipation", "toxic_look_(typhos)", "belly_pain"],
            "Tuberculosis": ["cough", "weight_loss", "sweating", "blood_in_sputum"],
            "Common Cold": ["continuous_sneezing", "chills", "throat_irritation", "runny_nose", "congestion"],
            "Pneumonia": ["cough", "breathlessness", "phlegm", "fast_heart_rate", "rusty_sputum"],
            "Acne": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
            
            # --- Custom Diseases (India-specific / emerging) ---
            "COVID-19": ["high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness"],
            "Monkeypox": ["high_fever", "skin_rash", "swelled_lymph_nodes", "headache", "muscle_pain"],
            "Heatstroke": ["high_fever", "dizziness", "fast_heart_rate", "headache"],
            "Polio": ["muscle_weakness", "weakness_in_limbs", "high_fever", "fatigue"]
        }
        
        self.load_rules(rules_path)

    def load_rules(self, rules_path: str = None):
        """Load rules from external JSON or fall back to default knowledge base."""
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                loaded = json.load(f)
                for disease, details in loaded.items():
                    if "required" in details:
                        self.rules[disease] = details["required"]
        else:
            self.rules = self.default_rules

    def evaluate(self, active_symptoms: List[str], threshold: float = 0.5) -> List[RuleMatch]:
        """
        Executes Forward Chaining over the rule knowledge base.

        Forward Chaining Algorithm:
        1. KNOWN FACTS = set(active_symptoms)
        2. FOR each rule R in knowledge base:
              required = set(R.symptoms)
              overlap = KNOWN FACTS ∩ required
              match_ratio = |overlap| / |required|
              IF match_ratio >= threshold THEN fire rule → add diagnosis
        3. SORT all fired diagnoses by match_strength DESC
        4. RETURN ranked candidate list

        :param active_symptoms: List of symptom string keys reported by patient
        :param threshold: Minimum match ratio to fire a rule (0.5 = 50% overlap)
        :return: List of RuleMatch instances sorted by match strength descending
        """
        matches = []
        # Step 1: Establish known facts from patient input
        patient_facts = set(active_symptoms)
        
        # Step 2: Forward chain through all rules
        for disease, required_symptoms in self.rules.items():
            rule_set = set(required_symptoms)
            
            if not rule_set:
                continue
            
            # Compute fact-rule overlap (Forward Chaining match)
            overlapped = patient_facts.intersection(rule_set)
            match_ratio = len(overlapped) / len(rule_set)
            
            # Step 3: Fire rule if threshold met
            if match_ratio >= threshold:
                matches.append(RuleMatch(
                    disease=disease,
                    match_strength=round(match_ratio, 3),
                    matched_symptoms=list(overlapped)
                ))
                
        # Step 4: Rank all fired rules by confidence
        matches.sort(key=lambda x: x.match_strength, reverse=True)
        return matches

if __name__ == "__main__":
    eng = RuleEngine()
    test_symptoms = ["vomiting", "diarrhoea", "sunken_eyes", "dehydration"]
    res = eng.evaluate(test_symptoms)
    for r in res:
        print(f"Matched {r.disease} with strength {r.match_strength}")
