serious = [
    "Heart Attack", "Pneumonia", "Tuberculosis", "Hepatitis B",
    "Hepatitis C", "Malaria", "Dengue", "Paralysis (brain hemorrhage)",
    "Jaundice", "AIDS"
]

moderate = [
    "Typhoid", "Chicken Pox", "Diabetes", "Hypertension",
    "Bronchial Asthma", "Urinary Tract Infection", "Psoriasis",
    "Gastroenteritis", "Hypoglycemia", "Hyperthyroidism", "Hypothyroidism"
]

warning_text = "This is a preliminary screening tool. Consult a registered medical practitioner for accurate diagnosis."


def get_urgency(disease, confidence):
    if disease in serious and confidence > 30:
        return "HIGH"
    elif disease in moderate and confidence > 20:
        return "MODERATE"
    return "LOW"
