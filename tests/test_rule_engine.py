import unittest
import json
from src.rule_engine import RuleEngine

class TestRules(unittest.TestCase):
    def setUp(self):
        self.engine = RuleEngine()
        with open("data/processed/symptom_list.json") as f:
            self.symptoms = json.load(f)

    def make_vec(self, names):
        vec = [0] * len(self.symptoms)
        for s in names:
            if s in self.symptoms:
                vec[self.symptoms.index(s)] = 1
        return vec

    def test_malaria_full_match(self):
        vec = self.make_vec(["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"])
        out = self.engine.evaluate(vec)
        self.assertIn("Malaria", out)
        self.assertEqual(out["Malaria"]["confidence"], 100.0)

    def test_empty_gives_nothing(self):
        out = self.engine.evaluate([0] * len(self.symptoms))
        self.assertEqual(len(out), 0)

    def test_partial_never_100(self):
        vec = self.make_vec(["high_fever", "headache"])
        out = self.engine.evaluate(vec)
        for d in out.values():
            self.assertLess(d["confidence"], 100)

    def test_malaria_ranks_first(self):
        vec = self.make_vec(["chills", "high_fever", "sweating", "headache", "muscle_pain"])
        out = self.engine.evaluate(vec)
        self.assertEqual(list(out.keys())[0], "Malaria")

if __name__ == "__main__":
    unittest.main()
