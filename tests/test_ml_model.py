import unittest
import os
from src.ml_model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.ml_model = MLModel()
        # Ensure model is trained for tests
        if self.ml_model.model is None:
            self.ml_model.train()
            
    def test_model_loading(self):
        self.assertIsNotNone(self.ml_model.model)
        self.assertTrue(len(self.ml_model.disease_list) > 0)
        self.assertTrue(len(self.ml_model.symptom_list) > 0)
        
    def test_predict_proba_format(self):
        symptoms = ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"]
        probs = self.ml_model.predict_proba(symptoms)
        
        self.assertTrue(len(probs) > 0)
        
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)
        
    def test_predict_disease(self):
        symptoms = ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"]
        probs = self.ml_model.predict_proba(symptoms)
        
        # Gastroenteritis should be high prediction
        highest_disease = list(probs.keys())[0]
        # In this dataset, this combination maps strongly to Gastroenteritis
        self.assertEqual(highest_disease, "Gastroenteritis")

if __name__ == "__main__":
    unittest.main()
