import unittest
from src.rule_engine import RuleEngine, RuleMatch

class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RuleEngine()

    def test_exact_match(self):
        symptoms = ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"]
        matches = self.engine.evaluate(symptoms)
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0].disease, "Gastroenteritis")
        self.assertEqual(matches[0].match_strength, 1.0)
        
    def test_partial_match(self):
        symptoms = ["vomiting", "sunken_eyes"]
        matches = self.engine.evaluate(symptoms, threshold=0.5)
        # 2 out of 4 is 0.5, so Gastroenteritis should match
        self.assertTrue(any(m.disease == "Gastroenteritis" for m in matches))
        
    def test_no_match(self):
        symptoms = ["itching", "shivering"]
        # With threshold 0.8, these disjoint symptoms shouldn't match any single disease strongly
        matches = self.engine.evaluate(symptoms, threshold=0.8)
        self.assertEqual(len(matches), 0)
        
    def test_multiple_matches(self):
        symptoms = ["continuous_sneezing", "chills", "fatigue", "high_fever", "shivering"]
        # Common cold and Allergy share continuous_sneezing, chills, etc.
        matches = self.engine.evaluate(symptoms, threshold=0.2)
        diseases = [m.disease for m in matches]
        self.assertIn("Allergy", diseases)
        self.assertIn("Common Cold", diseases)
        
if __name__ == "__main__":
    unittest.main()
