"""
MedAssist AI — Confidence Fusion Layer

Implements the weighted confidence fusion formula from the system architecture:
    Final Score = (w₁ × Rule Score) + (w₂ × ML Score)

This layer bridges symbolic AI (interpretability) with probabilistic ML
(pattern-recognition), producing a single ranked diagnosis list.

Default weights: Rule = 0.4, ML = 0.6
"""

from typing import List, Dict, Any

class FusionEngine:
    """
    Confidence Aggregator that merges rule-based and ML outputs.

    Fusion Formula:
        confidence(disease) = (rule_weight × rule_strength) + (ml_weight × ml_probability)

    This weighted scheme ensures:
    - Rules provide medically grounded, interpretable baseline
    - ML adds statistical pattern-recognition over 132-symptom feature space
    - Neither system alone dominates the final prediction
    """
    def __init__(self, rule_weight: float = 0.4, ml_weight: float = 0.6):
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight

    def fuse(self, rule_results: List[Any], ml_results: Dict[str, float], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Fuses rule engine and ML model outputs into a unified ranked list.

        :param rule_results: List of RuleMatch items from Forward Chaining engine
        :param ml_results: Dict from ML predict_proba mapping disease to probability
        :param top_n: Number of top predictions to return
        :return: Sorted list of fused results up to top_n
        """
        combined = {}

        # 1. Process Rule Engine (Forward Chaining) Results
        for match in rule_results:
            combined[match.disease] = {
                "disease": match.disease,
                "rule_strength": match.match_strength,
                "ml_probability": 0.0,
                "matched_symptoms": match.matched_symptoms
            }

        # 2. Process ML (Random Forest) Results
        for disease, prob in ml_results.items():
            if disease in combined:
                combined[disease]["ml_probability"] = prob
            else:
                combined[disease] = {
                    "disease": disease,
                    "rule_strength": 0.0,
                    "ml_probability": prob,
                    "matched_symptoms": []
                }

        # 3. Apply Weighted Fusion Formula
        fused_results = []
        for disease, data in combined.items():
            # If disease is ONLY in Rule Engine (no ML training data for it)
            if disease not in ml_results:
                final_score = data["rule_strength"]  # 100% Rule Reliant
                rule_contrib = final_score
                ml_contrib = 0.0
            else:
                # Weighted fusion: Final = (w1 × Rule) + (w2 × ML)
                final_score = (self.rule_weight * data["rule_strength"]) + (self.ml_weight * data["ml_probability"])
                rule_contrib = self.rule_weight * data["rule_strength"]
                ml_contrib = self.ml_weight * data["ml_probability"]
                
            fused_results.append({
                "disease": disease,
                "confidence": round(final_score, 4),
                "rule_contribution": round(rule_contrib, 4),
                "ml_contribution": round(ml_contrib, 4),
                "matched_symptoms": data["matched_symptoms"]
            })

        # 4. Rank by fused confidence score (descending)
        fused_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return fused_results[:top_n]
