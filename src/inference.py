import json
import os
from src.rule_engine import RuleEngine
from src.ml_model import DiseasePredictor
from src.fusion import combine_scores
from src.triage import get_urgency, warning_text

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class Pipeline:
    def __init__(self):
        self.rules = RuleEngine()
        self.ml = DiseasePredictor()
        self.ml.load()
        with open(os.path.join(data_path, "symptom_list.json")) as f:
            self.symptoms = json.load(f)

    def run(self, user_symptoms, top_k=5):
        rule_hits = self.rules.evaluate(user_symptoms)
        ml_hits = self.ml.predict(user_symptoms)
        scores = combine_scores(rule_hits, ml_hits)

        results = []
        for disease, conf in list(scores.items())[:top_k]:
            matched = rule_hits[disease]["matched_symptoms"] if disease in rule_hits else []
            results.append({
                "disease": disease,
                "confidence": conf,
                "urgency": get_urgency(disease, conf),
                "matched_symptoms": matched
            })

        return {
            "predictions": results,
            "rule_matches": rule_hits,
            "disclaimer": warning_text
        }
