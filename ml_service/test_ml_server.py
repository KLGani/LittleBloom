import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_server


class PredictionEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = ml_server.app.test_client()

    def test_both_models_accept_their_complete_feature_contracts(self):
        examples = {
            "pregnancy": {
                "systolicBP_mmHg": 120,
                "diastolicBP_mmHg": 80,
                "heartRate_bpm": 80,
                "temperature_C": 37.0,
                "bloodSugar_mg_dL": 90,
                "recordWeight_kg": 65,
                "sleepHours": 8,
            },
            "child": {
                "ageMonths_derived": 59,
                "systolicBP_mmHg": 116,
                "diastolicBP_mmHg": 99,
                "heartRate_bpm": 105,
                "temperature_C": 36.2,
                "bloodSugar_mg_dL": 92.1,
                "recordWeight_kg": 18.8,
                "sleepHours": 9.7,
            },
        }

        for mode, features in examples.items():
            with self.subTest(mode=mode):
                response = self.client.post(
                    "/predict-alert",
                    json={"mode": mode, "features": features},
                )

                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertTrue(result["model_assessed"])
                self.assertIsInstance(result["model_requires_attention"], bool)
                self.assertIsInstance(result["requires_attention"], bool)

    def test_rejects_missing_features_and_non_finite_values(self):
        missing = self.client.post(
            "/predict-alert",
            json={"mode": "child", "features": {}},
        )
        non_finite = self.client.post(
            "/predict-alert",
            json={
                "mode": "pregnancy",
                "features": {
                    "systolicBP_mmHg": 120,
                    "diastolicBP_mmHg": 80,
                    "heartRate_bpm": 80,
                    "temperature_C": 37.0,
                    "bloodSugar_mg_dL": 90,
                    "recordWeight_kg": 65,
                    "sleepHours": "nan",
                },
            },
        )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(non_finite.status_code, 400)

    def test_health_endpoint_reports_readiness(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")


if __name__ == "__main__":
    unittest.main()
