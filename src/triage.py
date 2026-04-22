"""
MedAssist AI — Triage Engine

Classifies urgency levels based on fused confidence and disease severity.
Provides actionable next steps without suggesting treatments or medications.

Triage Levels:
- Low: Non-critical, monitor symptoms
- Moderate: Medical consultation recommended
- High: Urgent attention required

Ethical Guardrails:
- Does NOT trigger emergency services
- Does NOT suggest treatments or medications
- Does NOT replace clinical judgment
"""

from typing import Dict, Any

class TriageEngine:
    """
    Risk assessment engine that classifies urgency based on:
    1. Fused confidence score from the Confidence Aggregator
    2. Disease severity classification (critical vs non-critical)

    All outputs include recommended next steps (action plans)
    without crossing into treatment recommendation territory.
    """
    def __init__(self):
        # High-urgency conditions based on WHO/ICMR severity classifications
        self.critical_conditions = {
            "Heart attack", 
            "Tuberculosis", 
            "Dengue", 
            "Typhoid", 
            "Pneumonia", 
            "Malaria",
            "Bronchial Asthma",
            "COVID-19",
            "Monkeypox",
            "Heatstroke",
            "Paralysis (brain hemorrhage)"
        }

    def assess(self, top_prediction: Dict[str, Any]) -> Dict[str, str]:
        """
        Assess triage urgency from the top fused prediction.

        Decision Logic:
        - High: confidence >= 45% AND disease is in critical_conditions
        - Moderate: confidence >= 30%
        - Low: confidence < 30% or no strong match

        :param top_prediction: The top FusedResult dictionary
        :return: Triage dict with level, message, and action_plan
        """
        if not top_prediction:
            return {
                "level": "Low", 
                "message": "Symptoms do not map strongly to known critical conditions.",
                "action_plan": "No specific action required based on inputs. Re-evaluate if symptoms change or persist."
            }

        disease = top_prediction["disease"]
        confidence = top_prediction["confidence"]

        if confidence >= 0.45 and disease in self.critical_conditions:
            return {
                "level": "High",
                "message": "Immediate medical consultation recommended due to nature of symptoms.",
                "action_plan": "Seek emergency medical attention or consult a specialist immediately. Avoid self-medication."
            }
        elif confidence >= 0.30:
            return {
                "level": "Moderate",
                "message": "Consider consulting a healthcare professional soon.",
                "action_plan": "Schedule an appointment with a primary care physician. Rest and monitor symptoms closely."
            }
        else:
            return {
                "level": "Low",
                "message": "No urgent warnings identified, but monitor your symptoms.",
                "action_plan": "Rest, maintain hydration, and observe for 24-48 hours. Consult a doctor if symptoms worsen."
            }
