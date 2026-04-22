"""
MedAssist AI — Autonomous Medical Triage Agent

This module implements the core Agentic AI architecture:

    Perception → Reasoning → Action

The agent autonomously:
1. PERCEIVES patient symptoms (input parsing)
2. REASONS using two parallel AI tools:
   - Tool 1: Forward Chaining Rule Engine (symbolic, deterministic)
   - Tool 2: Random Forest ML Classifier (probabilistic, data-driven)
3. SYNTHESIZES findings via Confidence Fusion (weighted aggregation)
4. ACTS by generating:
   - Top 3 ranked diagnoses with confidence scores
   - Transparent reasoning explanation (why this diagnosis)
   - Triage-based action plan (what to do next)
   - Mandatory ethical disclaimer

Agentic AI Concepts Demonstrated:
- Tool Use: Agent orchestrates multiple specialized AI tools
- Autonomous Reasoning: Agent generates its own explanation chain
- Goal-Directed Behavior: Agent's goal is accurate triage, not just prediction
- Guardrails: Agent enforces ethical boundaries (disclaimer, no treatment advice)
"""

from typing import List, Dict, Any
from src.rule_engine import RuleEngine
from src.ml_model import MLModel
from src.fusion import FusionEngine
from src.triage import TriageEngine

class MedicalTriageAgent:
    """
    Autonomous Medical Triage Agent.

    Architecture (Agentic AI Pattern):
    ┌──────────────────────────────────────────────┐
    │              PERCEPTION LAYER                │
    │  Symptom Input → Binary Feature Vector       │
    └──────────────────┬───────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────┐
    │              TOOL ORCHESTRATION              │
    │  ┌─────────────┐    ┌──────────────────┐     │
    │  │ Tool 1:     │    │ Tool 2:          │     │
    │  │ Rule Engine │    │ ML Classifier    │     │
    │  │ (Forward    │    │ (Random Forest)  │     │
    │  │  Chaining)  │    │                  │     │
    │  └──────┬──────┘    └────────┬─────────┘     │
    │         └──────┬─────────────┘               │
    │       ┌────────▼──────────┐                  │
    │       │ Confidence Fusion │                  │
    │       └────────┬──────────┘                  │
    └────────────────┼─────────────────────────────┘
                     │
    ┌────────────────▼─────────────────────────────┐
    │              ACTION LAYER                    │
    │  Reasoning + Triage + Action Plan            │
    │  + Ethical Guardrails (Disclaimer)            │
    └──────────────────────────────────────────────┘
    """
    def __init__(self):
        # Register the agent's internal tools
        self.tools = {
            "rule_engine": RuleEngine(),
            "ml_classifier": MLModel(),
            "fusion": FusionEngine(rule_weight=0.4, ml_weight=0.6),
            "triage": TriageEngine()
        }
        
        # Ethical guardrail (mandatory on every output)
        self.disclaimer = (
            "This is a preliminary screening tool developed for academic purposes. "
            "It does not provide medical diagnoses. "
            "Consult a registered medical practitioner for accurate diagnosis."
        )

    @property
    def rule_tool(self):
        return self.tools["rule_engine"]

    @property
    def ml_tool(self):
        return self.tools["ml_classifier"]

    @property
    def fusion_tool(self):
        return self.tools["fusion"]

    @property
    def triage_tool(self):
        return self.tools["triage"]

    def _generate_reasoning(self, predictions: List[Dict], active_symptoms: List[str]) -> str:
        """
        Agentic Reasoning Chain Generation.

        The agent introspects its own tool outputs and constructs a
        human-readable explanation of its decision-making process.
        This is the "explainability" requirement from the architecture.
        """
        if not predictions:
            return (
                "I analyzed the reported symptoms but could not find confident "
                "diagnostic patterns across my rule base or ML models. "
                "The symptoms may be too general or not well-represented in my training data."
            )
        
        top = predictions[0]
        disease = top['disease']
        conf = top['confidence'] * 100
        r_conf = top['rule_contribution'] * 100
        m_conf = top['ml_contribution'] * 100
        
        reasoning = f"I analyzed {len(active_symptoms)} reported symptom(s) using my dual-inference tools. "
        
        # Explain which tools contributed
        if r_conf > 0 and m_conf > 0:
            reasoning += (
                f"Both tools detected signals for '{disease}'. "
                f"My Rule Engine (Forward Chaining) matched {r_conf:.1f}% of the diagnostic criteria, "
                f"while my ML Classifier (Random Forest) estimated a {m_conf:.1f}% probability "
                f"based on learned symptom co-occurrence patterns. "
            )
        elif r_conf > 0:
            reasoning += (
                f"My Rule Engine matched these symptoms to '{disease}' ({r_conf:.1f}%), "
                f"though my ML Classifier lacked confident probabilistic signals. "
            )
        elif m_conf > 0:
            reasoning += (
                f"My ML Classifier identified a {m_conf:.1f}% probabilistic match for '{disease}', "
                f"but the symptoms did not strongly trigger my deterministic rule base. "
            )
        
        reasoning += f"After confidence fusion (40% Rule + 60% ML weighting), the overall confidence is {conf:.1f}%. "
        
        # Explain matched symptoms if available
        if top.get('matched_symptoms'):
            pretty = [s.replace('_', ' ').title() for s in top['matched_symptoms']]
            reasoning += f"Key contributing symptoms from rules: {', '.join(pretty)}. "
        
        # Mention additional candidates if present
        if len(predictions) > 1:
            others = [f"{p['disease']} ({p['confidence']*100:.1f}%)" for p in predictions[1:]]
            reasoning += f"Alternative candidates considered: {', '.join(others)}."
        
        return reasoning

    def run(self, active_symptoms: List[str]) -> Dict[str, Any]:
        """
        Execute the full agentic workflow.

        Pipeline (matches Final Report's 6-step architecture):
        Step 1: Perception — Accept symptom inputs
        Step 2: Tool 1 — Forward Chaining Rule Engine
        Step 3: Tool 2 — Random Forest ML Classifier
        Step 4: Synthesis — Confidence Fusion (weighted aggregation)
        Step 5: Assessment — Triage classification
        Step 6: Output — Reasoning + Action Plan + Disclaimer

        :param active_symptoms: List of symptom string keys
        :return: Structured agent response payload
        """
        if not active_symptoms:
            return {
                "top_predictions": [],
                "triage": self.triage_tool.assess(None),
                "agent_reasoning": "I need symptom inputs to begin my analysis.",
                "disclaimer": self.disclaimer
            }

        # Step 2: Tool 1 — Forward Chaining Rule Engine
        rule_results = self.rule_tool.evaluate(active_symptoms)

        # Step 3: Tool 2 — Random Forest ML Classifier
        ml_results = self.ml_tool.predict_proba(active_symptoms)

        # Step 4: Confidence Fusion — Top 3 predictions (per Final Report spec)
        fused_results = self.fusion_tool.fuse(
            rule_results=rule_results, 
            ml_results=ml_results, 
            top_n=3
        )

        # Step 5: Triage Assessment (based on highest-confidence prediction)
        top_prediction = fused_results[0] if fused_results else None
        triage_info = self.triage_tool.assess(top_prediction)

        # Step 6: Generate Agentic Reasoning + Assemble Output
        reasoning = self._generate_reasoning(fused_results, active_symptoms)

        payload = {
            "top_predictions": fused_results,
            "triage": triage_info,
            "agent_reasoning": reasoning,
            "disclaimer": self.disclaimer
        }
        
        return payload

if __name__ == "__main__":
    import json
    agent = MedicalTriageAgent()
    res = agent.run(["vomiting", "sunken_eyes", "dehydration", "diarrhoea"])
    print(json.dumps(res, indent=2))
