import unittest
from src.fusion import combine_scores

class TestFusion(unittest.TestCase):
    def test_strong_rule_wins(self):
        rules = {"Malaria": {"confidence": 100.0}}
        ml = {"Malaria": 80.0, "Typhoid": 10.0}
        out = combine_scores(rules, ml)
        self.assertEqual(list(out.keys())[0], "Malaria")
        self.assertGreater(out["Malaria"], 90)

    def test_ml_only_works(self):
        out = combine_scores({}, {"Diabetes": 70.0, "Malaria": 20.0})
        self.assertEqual(list(out.keys())[0], "Diabetes")

    def test_empty_is_empty(self):
        self.assertEqual(len(combine_scores({}, {})), 0)

if __name__ == "__main__":
    unittest.main()
