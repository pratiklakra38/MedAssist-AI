import unittest
from src.fusion import FusionEngine
from src.rule_engine import RuleMatch
from src.inference import MedicalTriageAgent

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = FusionEngine(rule_weight=0.4, ml_weight=0.6)
        
    def test_fuse_logic(self):
        rule_res = [RuleMatch(disease="Dengue", match_strength=0.8, matched_symptoms=[])]
        ml_res = {"Dengue": 0.9, "Malaria": 0.2}
        
        # Expected Dengue score: (0.4 * 0.8)[0.32] + (0.6 * 0.9)[0.54] = 0.86
        # Expected Malaria score: (0.4 * 0.0)[0.0] + (0.6 * 0.2)[0.12] = 0.12
        fused = self.fusion.fuse(rule_res, ml_res)
        
        self.assertEqual(len(fused), 2)
        dengue_res = next(i for i in fused if i["disease"] == "Dengue")
        malaria_res = next(i for i in fused if i["disease"] == "Malaria")
        
        self.assertAlmostEqual(dengue_res["confidence"], 0.86, places=2)
        self.assertAlmostEqual(malaria_res["confidence"], 0.12, places=2)
        
    def test_fuse_exclusive_results(self):
        """When a disease is ONLY in rule results (no ML data), fusion uses 100% rule strength."""
        rule_res = [RuleMatch(disease="Gastroenteritis", match_strength=1.0, matched_symptoms=[])]
        ml_res = {"Malaria": 0.5}
        
        fused = self.fusion.fuse(rule_res, ml_res)
        self.assertEqual(len(fused), 2)
        
        gastro_res = next(i for i in fused if i["disease"] == "Gastroenteritis")
        # Rule-only disease gets full rule_strength (not weighted)
        self.assertAlmostEqual(gastro_res["confidence"], 1.0, places=2)

class TestMedicalTriageAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MedicalTriageAgent()
        
    def test_agent_integration(self):
        symptoms = ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"]
        res = self.agent.run(symptoms)
        
        # Ensure guardrails exist
        self.assertIn("preliminary screening tool", res["disclaimer"])
        
        # Ensure triage has level and action_plan
        self.assertIn(res["triage"]["level"], ["Low", "Moderate", "High"])
        self.assertIn("action_plan", res["triage"])
        
        # Ensure we have top predictions (up to 3) and agent reasoning
        self.assertTrue(len(res["top_predictions"]) > 0)
        self.assertTrue(len(res["top_predictions"]) <= 3)
        self.assertIn("agent_reasoning", res)
        
        top_pred = res["top_predictions"][0]
        self.assertEqual(top_pred["disease"], "Gastroenteritis")
        self.assertTrue("confidence" in top_pred)
        self.assertTrue("rule_contribution" in top_pred)
        self.assertTrue("ml_contribution" in top_pred)
        
    def test_agent_empty_input(self):
        res = self.agent.run([])
        self.assertEqual(res["top_predictions"], [])
        self.assertIn("agent_reasoning", res)

    def test_agent_tools_registered(self):
        """Verify the agent has all 4 tools registered."""
        self.assertIn("rule_engine", self.agent.tools)
        self.assertIn("ml_classifier", self.agent.tools)
        self.assertIn("fusion", self.agent.tools)
        self.assertIn("triage", self.agent.tools)

if __name__ == "__main__":
    unittest.main()
