"""
utils.py
Helper Utilities & Safe Data Processing for MediGuide AI

Provides safe JSON parsing, input validation, urgency color mapping,
and report export utilities (PDF & TXT).
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, Tuple
from fpdf import FPDF
from src.config import DEFAULT_PATIENT_NAME


def clean_and_parse_json(raw_response: str) -> Tuple[Dict[str, Any], bool, str]:
    """
    Safely cleans and parses JSON output from LLM response string.
    Handles surrounding text, markdown code blocks, and schema normalization gracefully.

    Returns:
        Tuple[Dict, bool, str]: (Parsed dict, Success flag, Error/Warning message if any)
    """
    if not raw_response:
        return _get_fallback_json("Empty response received from language model."), False, "Empty LLM output."

    cleaned = raw_response.strip()

    # Regex to extract JSON block inside ```json ... ``` or ``` ... ```
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
    if json_match:
        cleaned = json_match.group(1).strip()
    else:
        # Fallback: search for first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1].strip()

    try:
        data = json.loads(cleaned)
        normalized_data = _normalize_schema(data)
        return normalized_data, True, ""

    except json.JSONDecodeError as decode_err:
        error_msg = f"JSON decode error: {str(decode_err)}"
        fallback_data = _get_fallback_json(
            raw_text=raw_response,
            error_note="Failed to parse structured JSON output. Displaying raw LLM output below for inspection."
        )
        return fallback_data, False, error_msg


def _normalize_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all expected schema keys exist with appropriate fallback data types and clean urgency levels.
    """
    expected_schema = {
        "summary": "Symptom assessment summary unavailable.",
        "possible_conditions": [],
        "urgency_level": "MEDIUM",
        "recommended_next_steps": ["Monitor symptoms closely and get adequate rest.", "Arrange professional consultation if symptoms persist or worsen."],
        "questions_for_doctor": ["Could these symptoms be related to an underlying condition?", "Are any diagnostic tests appropriate?"],
        "warning_signs": ["Seek emergency care immediately if experiencing severe chest pain or shortness of breath."]
    }

    normalized = {}
    for key, default in expected_schema.items():
        val = data.get(key, default)
        if key == "urgency_level":
            urg_str = str(val).upper().strip()
            if "EMERGENCY" in urg_str:
                val = "EMERGENCY"
            elif "HIGH" in urg_str:
                val = "HIGH"
            elif "LOW" in urg_str:
                val = "LOW"
            else:
                val = "MEDIUM"
        normalized[key] = val

    # Standardize possible_conditions structure
    if isinstance(normalized["possible_conditions"], list):
        cleaned_conditions = []
        for item in normalized["possible_conditions"]:
            if isinstance(item, dict):
                cleaned_conditions.append({
                    "name": item.get("name", "Unspecified Illness Category"),
                    "reason": item.get("reason", "This symptom pattern may be associated with common physiological factors. This is not a diagnosis.")
                })
            elif isinstance(item, str):
                cleaned_conditions.append({
                    "name": item,
                    "reason": "Educational correlation • Not a diagnosis."
                })
        normalized["possible_conditions"] = cleaned_conditions
    else:
        normalized["possible_conditions"] = []

    return normalized


def _get_fallback_json(raw_text: str = "", error_note: str = "") -> Dict[str, Any]:
    """
    Returns a safe fallback dictionary if JSON parsing fails completely.
    """
    return {
        "summary": error_note or "Unable to generate structured summary.",
        "possible_conditions": [
            {
                "name": "General Symptom Pattern",
                "reason": "The model response did not produce valid JSON. Raw output is available in Technical Inspection."
            }
        ],
        "urgency_level": "MEDIUM",
        "recommended_next_steps": [
            "Monitor symptoms closely",
            "Maintain hydration and rest",
            "Consult a qualified healthcare professional"
        ],
        "questions_for_doctor": [
            "Could these symptoms be related to an underlying condition?",
            "Are any diagnostic tests appropriate?"
        ],
        "warning_signs": [
            "Severe difficulty breathing or chest pain",
            "High persistent fever",
            "Sudden severe weakness or fainting"
        ],
        "raw_response": raw_text
    }


def validate_symptom_inputs(selected_symptoms: list, free_text_symptoms: str) -> Tuple[bool, str]:
    """
    Validates that the user has provided at least one symptom.
    If empty, returns False and friendly error message per rubric spec.
    """
    has_multiselect = bool(selected_symptoms)
    has_freetext = bool(free_text_symptoms and free_text_symptoms.strip())

    if not has_multiselect and not has_freetext:
        return False, "Please provide at least one symptom before starting the assessment."
    return True, ""


def get_urgency_badge_details(urgency_level: str) -> Dict[str, str]:
    """
    Maps urgency levels (LOW, MEDIUM, HIGH, EMERGENCY) to dark clinical UI colors and descriptions.
    """
    urgency = str(urgency_level).upper().strip()

    if urgency == "LOW":
        return {
            "label": "🟢 LOW URGENCY",
            "color_class": "success",
            "badge_color": "#10b981",
            "bg_color": "#064e3b",
            "border_color": "#059669",
            "description": "Routine self-monitoring may be appropriate."
        }
    elif urgency == "MEDIUM":
        return {
            "label": "🟡 MEDIUM URGENCY",
            "color_class": "warning",
            "badge_color": "#f59e0b",
            "bg_color": "#78350f",
            "border_color": "#d97706",
            "description": "Non-urgent professional healthcare consultation is recommended, particularly if symptoms persist or worsen."
        }
    elif urgency == "HIGH":
        return {
            "label": "🟠 HIGH URGENCY",
            "color_class": "error",
            "badge_color": "#f97316",
            "bg_color": "#7c2d12",
            "border_color": "#ea580c",
            "description": "Prompt medical evaluation is recommended."
        }
    elif urgency == "EMERGENCY":
        return {
            "label": "🔴 EMERGENCY",
            "color_class": "critical",
            "badge_color": "#ef4444",
            "bg_color": "#7f1d1d",
            "border_color": "#dc2626",
            "description": "Seek emergency medical attention immediately!"
        }
    else:
        return {
            "label": "🟡 MEDIUM URGENCY",
            "color_class": "info",
            "badge_color": "#f59e0b",
            "bg_color": "#78350f",
            "border_color": "#d97706",
            "description": "Consult a healthcare provider for clinical evaluation."
        }


def format_patient_inputs(
    patient_name: str,
    age: str,
    gender: str,
    symptoms_list: list,
    custom_symptoms: str,
    duration: str,
    severity: int,
    conditions: str,
    medications: str,
    notes: str,
    language: str
) -> Dict[str, Any]:
    """
    Formats raw Streamlit widget inputs into a clean dictionary for LangChain prompts.
    """
    all_symptoms = list(symptoms_list)
    if custom_symptoms and custom_symptoms.strip():
        all_symptoms.append(custom_symptoms.strip())
    
    symptoms_str = ", ".join(all_symptoms) if all_symptoms else "None specified"
    clean_name = patient_name.strip() if patient_name and patient_name.strip() else DEFAULT_PATIENT_NAME

    return {
        "patient_name": clean_name,
        "age": age if age and age.strip() else "Not provided",
        "gender": gender,
        "symptoms": symptoms_str,
        "duration": duration,
        "severity": severity,
        "existing_conditions": conditions.strip() if conditions and conditions.strip() else "None reported",
        "medications": medications.strip() if medications and medications.strip() else "None reported",
        "additional_notes": notes.strip() if notes and notes.strip() else "None provided",
        "language": language
    }


def generate_text_report(patient_data: Dict[str, Any], assessment: Dict[str, Any], narrative_text: str = "") -> str:
    """
    Generates a clean text report for downloading.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "============================================================",
        "             MEDIGUIDE AI - PATIENT GUIDANCE REPORT         ",
        "============================================================",
        f"Date/Time: {now_str}",
        "DISCLAIMER: Educational prototype only. NOT a confirmed diagnosis.",
        "Consult a qualified healthcare provider for clinical medical advice.",
        "------------------------------------------------------------",
        "",
        "--- PATIENT INFORMATION ---",
        f"Patient Name: {patient_data.get('patient_name')}",
        f"Age: {patient_data.get('age')}",
        f"Gender: {patient_data.get('gender')}",
        f"Symptoms: {patient_data.get('symptoms')}",
        f"Duration: {patient_data.get('duration')}",
        f"Severity: {patient_data.get('severity')}/10",
        f"Existing Medical Conditions: {patient_data.get('existing_conditions')}",
        f"Current Medications: {patient_data.get('medications')}",
        f"Additional Notes: {patient_data.get('additional_notes')}",
        "",
        "--- ASSESSMENT STATUS ---",
        f"Assessed Urgency Level: {assessment.get('urgency_level')}",
        f"Summary: {assessment.get('summary')}",
        "",
        "--- POTENTIAL CLINICAL CORRELATIONS (Educational Only) ---"
    ]

    for cond in assessment.get("possible_conditions", []):
        lines.append(f"• {cond.get('name')}: {cond.get('reason')}")

    lines.extend(["", "--- RECOMMENDED NEXT STEPS ---"])
    for step in assessment.get("recommended_next_steps", []):
        lines.append(f"• {step}")

    lines.extend(["", "--- QUESTIONS FOR YOUR DOCTOR ---"])
    for q in assessment.get("questions_for_doctor", []):
        lines.append(f"• {q}")

    lines.extend(["", "--- RED-FLAG WARNING SIGNS ---"])
    for w in assessment.get("warning_signs", []):
        lines.append(f"• {w}")

    if narrative_text:
        lines.extend(["", "--- AI PATIENT GUIDANCE NARRATIVE ---", narrative_text])

    lines.extend([
        "",
        "------------------------------------------------------------",
        "Generated by MediGuide AI Prototype",
        "============================================================"
    ])

    return "\n".join(lines)


def generate_pdf_report(patient_data: Dict[str, Any], assessment: Dict[str, Any], narrative_text: str = "") -> bytes:
    """
    Generates a PDF report using fpdf2.
    """
    pdf = FPDF()
    pdf.add_page()
    epw = pdf.epw
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(epw, 10, "MediGuide AI - Patient Guidance Report", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(epw, 5, f"Generated on {now_str} | Educational Prototype Only - Not a Confirmed Diagnosis", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Patient info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(epw, 7, "Patient Information:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(epw, 5, f"Patient Name: {patient_data.get('patient_name')}  |  Age: {patient_data.get('age')}  |  Gender: {patient_data.get('gender')}")
    pdf.multi_cell(epw, 5, f"Symptoms: {patient_data.get('symptoms')}")
    pdf.multi_cell(epw, 5, f"Duration: {patient_data.get('duration')}  |  Severity: {patient_data.get('severity')}/10")
    pdf.multi_cell(epw, 5, f"Medical Conditions: {patient_data.get('existing_conditions')}")
    pdf.multi_cell(epw, 5, f"Medications: {patient_data.get('medications')}")
    pdf.ln(4)

    # Urgency Level
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(epw, 7, f"Assessed Urgency Level: {assessment.get('urgency_level')}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(epw, 5, f"Summary: {assessment.get('summary')}")
    pdf.ln(4)

    # Possible Conditions
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(epw, 6, "Potential Clinical Correlations (Educational Only):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for cond in assessment.get("possible_conditions", []):
        pdf.multi_cell(epw, 5, f"- {cond.get('name')}: {cond.get('reason')}")
    pdf.ln(3)

    # Next Steps
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(epw, 6, "Recommended Next Steps:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for step in assessment.get("recommended_next_steps", []):
        pdf.multi_cell(epw, 5, f"- {step}")
    pdf.ln(3)

    # Questions for doctor
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(epw, 6, "Questions for Your Doctor:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for q in assessment.get("questions_for_doctor", []):
        pdf.multi_cell(epw, 5, f"- {q}")
    pdf.ln(3)

    # Warning Signs
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(epw, 6, "Red-Flag Warning Signs:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for w in assessment.get("warning_signs", []):
        pdf.multi_cell(epw, 5, f"- {w}")

    return bytes(pdf.output())
