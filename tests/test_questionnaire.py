import unittest

from app import app
from Ai.patient_memory import delete_patient_memory, get_patient_memory


VALID_ANSWERS = {
    "age": "34",
    "sex_assigned_at_birth": "female",
    "natural_skin_tone": "fair_pale",
    "previous_bp_diagnosis": "no",
    "recent_bp_reading": "yes",
    "bp_systolic_value": "120",
    "bp_diastolic_value": "80",
    "bp_medications": "no",
    "bp_symptoms": "no",
    "family_bp_history": "not_sure",
    "physical_activity": "few_times_week",
    "sodium_intake": "moderate",
    "smoking_status": "never",
    "fasting_status": "no",
    "cold_hands_feet": "no",
}


class QuestionnaireRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_valid_answers_are_stored_in_the_session_schema(self):
        response = self.client.post(
            "/questionnaire",
            json={"answers": VALID_ANSWERS},
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])

        patient_id = body["patient_id"]
        self.addCleanup(delete_patient_memory, patient_id)
        patient = get_patient_memory(patient_id)

        self.assertIsNotNone(patient)
        self.assertEqual(patient["data"]["age_years"], 34)
        self.assertEqual(patient["data"]["hypertension_diagnosis"], "no")
        self.assertEqual(patient["data"]["systolic_bp_recent"], 120)

    def test_missing_conditional_readings_are_stored_as_none(self):
        answers = dict(VALID_ANSWERS, recent_bp_reading="no")
        answers.pop("bp_systolic_value")
        answers.pop("bp_diastolic_value")

        response = self.client.post(
            "/questionnaire",
            json={"answers": answers},
        )

        self.assertEqual(response.status_code, 200)
        patient_id = response.get_json()["patient_id"]
        self.addCleanup(delete_patient_memory, patient_id)
        patient = get_patient_memory(patient_id)

        self.assertIsNotNone(patient)
        self.assertIsNone(patient["data"]["systolic_bp_recent"])
        self.assertIsNone(patient["data"]["diastolic_bp_recent"])

    def test_invalid_age_and_non_object_payload_are_rejected(self):
        response = self.client.post(
            "/questionnaire",
            json={"answers": dict(VALID_ANSWERS, age="0")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

        response = self.client.post("/questionnaire", json=[])
        self.assertEqual(response.status_code, 400)


class FaqsRouteTests(unittest.TestCase):
    def test_faqs_page_is_available_and_selected_in_navigation(self):
        response = app.test_client().get("/faqs")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<h1>FAQs</h1>", response.data)
        self.assertIn(b"data-current-index=\"3\"", response.data)
        self.assertIn(b"aria-current=\"page\">FAQs</a>", response.data)
        self.assertEqual(response.data.count(b"class=\"faq-card\""), 10)


if __name__ == "__main__":
    unittest.main()
