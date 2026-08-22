"""
test_mediguide.py
Unit tests for MediGuide AI core utilities, cache management, JSON parsing, and report generation.
"""

import unittest
import os
from src.utils import (
    clean_and_parse_json,
    validate_symptom_inputs,
    get_urgency_badge_details,
    format_patient_inputs,
    generate_text_report,
    generate_pdf_report,
)
from src.cache_manager import configure_cache, get_cache_info


class TestMediGuideCore(unittest.TestCase):

    def test_clean_and_parse_json_valid(self):
        valid_json_str = """
        {
            "summary": "Patient has mild cough.",
            "possible_conditions": [{"name": "Common Cold", "reason": "Cough present"}],
            "urgency_level": "LOW",
            "recommended_next_steps": ["Rest"],
            "questions_for_doctor": ["How long will it last?"],
            "warning_signs": ["Fever above 102F"]
        }
        """
        data, ok, err = clean_and_parse_json(valid_json_str)
        self.assertTrue(ok)
        self.assertEqual(data["urgency_level"], "LOW")
        self.assertEqual(data["summary"], "Patient has mild cough.")
        self.assertEqual(len(data["possible_conditions"]), 1)

    def test_clean_and_parse_json_markdown_wrapped(self):
        markdown_json = """```json
        {
            "summary": "Patient reports severe chest pain.",
            "possible_conditions": [{"name": "Angina", "reason": "Chest pressure"}],
            "urgency_level": "EMERGENCY",
            "recommended_next_steps": ["Call 911 immediately"],
            "questions_for_doctor": ["Is it heart related?"],
            "warning_signs": ["Radiating arm pain"]
        }
        ```"""
        data, ok, err = clean_and_parse_json(markdown_json)
        self.assertTrue(ok)
        self.assertEqual(data["urgency_level"], "EMERGENCY")
        self.assertEqual(data["summary"], "Patient reports severe chest pain.")

    def test_clean_and_parse_json_malformed_fallback(self):
        invalid_json = "This is not valid json text at all!"
        data, ok, err = clean_and_parse_json(invalid_json)
        self.assertFalse(ok)
        self.assertIn("urgency_level", data)
        self.assertEqual(data["urgency_level"], "MEDIUM")

    def test_validate_symptom_inputs(self):
        valid_1, _ = validate_symptom_inputs(["Fever"], "")
        self.assertTrue(valid_1)

        valid_2, _ = validate_symptom_inputs([], "Headache and nausea")
        self.assertTrue(valid_2)

        invalid, msg = validate_symptom_inputs([], "   ")
        self.assertFalse(invalid)
        self.assertIn("at least one symptom", msg)

    def test_get_urgency_badge_details(self):
        low_badge = get_urgency_badge_details("LOW")
        self.assertIn("LOW URGENCY", low_badge["label"])

        emergency_badge = get_urgency_badge_details("EMERGENCY")
        self.assertIn("EMERGENCY", emergency_badge["label"])

    def test_format_patient_inputs(self):
        patient_dict = format_patient_inputs(
            patient_name="Jane Doe",
            age="45",
            gender="Female",
            symptoms_list=["Fever", "Cough"],
            custom_symptoms="Sore throat",
            duration="1 - 3 days",
            severity=6,
            conditions="Hypertension",
            medications="Lisinopril",
            notes="None",
            language="English",
        )
        self.assertEqual(patient_dict["patient_name"], "Jane Doe")
        self.assertEqual(patient_dict["age"], "45")
        self.assertIn("Fever", patient_dict["symptoms"])
        self.assertIn("Sore throat", patient_dict["symptoms"])

    def test_configure_cache(self):
        ok_mem, msg_mem = configure_cache("InMemoryCache")
        self.assertTrue(ok_mem)
        info_mem = get_cache_info("InMemoryCache")
        self.assertEqual(info_mem["name"], "InMemoryCache")

        ok_sqlite, msg_sqlite = configure_cache("SQLiteCache")
        self.assertTrue(ok_sqlite)
        info_sqlite = get_cache_info("SQLiteCache")
        self.assertEqual(info_sqlite["name"], "SQLiteCache")

        ok_off, msg_off = configure_cache("Disabled")
        self.assertTrue(ok_off)

    def test_generate_text_and_pdf_reports(self):
        patient_data = {
            "patient_name": "John Smith",
            "age": "30",
            "gender": "Male",
            "symptoms": "Headache",
            "duration": "1 day",
            "severity": 4,
            "existing_conditions": "None",
            "medications": "None",
        }
        assessment = {
            "urgency_level": "LOW",
            "summary": "Mild headache assessment.",
            "possible_conditions": [{"name": "Tension Headache", "reason": "Stress"}],
            "recommended_next_steps": ["Rest and hydrate"],
            "questions_for_doctor": ["How often does this occur?"],
            "warning_signs": ["Vision changes"],
        }
        txt = generate_text_report(patient_data, assessment)
        self.assertIn("MEDIGUIDE AI", txt)
        self.assertIn("John Smith", txt)
        self.assertIn("Tension Headache", txt)

        pdf_bytes = generate_pdf_report(patient_data, assessment)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 0)


if __name__ == "__main__":
    unittest.main()
