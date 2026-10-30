import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import unittest
from fastapi.testclient import TestClient
from api.main import app, startup_event

class TestFastAPIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicitly initialize state for test suite
        startup_event()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_classify_endpoint(self):
        payload = {
            "title": "System for Digital Packet Transmission in Wireless Networks",
            "abstract": "An apparatus configured for latency reduction and bandwidth allocation.",
            "independent_claims": "1. An apparatus comprising a network node processor."
        }
        response = self.client.post("/api/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("top_predicted_cpcs", data)
        self.assertIn("primary_cpc", data)

    def test_search_endpoint(self):
        payload = {
            "title": "System for Digital Packet Transmission in Wireless Networks",
            "abstract": "An apparatus configured for latency reduction and bandwidth allocation.",
            "independent_claims": "1. An apparatus comprising a network node processor.",
            "grant_date": "2025-01-01"
        }
        response = self.client.post("/api/search?top_k=5", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertLessEqual(len(data["results"]), 5)

if __name__ == "__main__":
    unittest.main()
