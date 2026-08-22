"""
config.py
MediGuide AI Configuration & Application Options

Contains constant definitions, medical safety disclaimers, form dropdown options,
and environment setup for OpenAI integration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# App Metadata
APP_NAME = "MediGuide AI"
APP_SUBTITLE = "AI-Powered Medical Symptom Assessment & Patient Guidance Assistant"
APP_VERSION = "1.0.0"

# Non-Negotiable Medical Safety Disclaimers
MEDICAL_DISCLAIMER_HEADER = (
    "⚠️ **IMPORTANT MEDICAL & SAFETY NOTICE**\n\n"
    "MediGuide AI is an **educational AI prototype only**. It is **NOT** a licensed medical provider, "
    "does **NOT** provide a confirmed medical diagnosis, and cannot replace professional clinical judgment "
    "or emergency services. Always consult a qualified healthcare provider for medical advice."
)

MEDICAL_DISCLAIMER_SIDEBAR = (
    "🛡️ **Educational Prototype Only**\n\n"
    "This system uses AI to generate preliminary guidance based on user input. "
    "It is not a substitute for a doctor. If you are experiencing a medical emergency, "
    "call your local emergency number (e.g. 911 or 112) immediately."
)

EMERGENCY_WARNING_BANNER = (
    "🚨 **IF YOU ARE EXPERIENCING A MEDICAL EMERGENCY** (such as sudden severe chest pain, "
    "extreme difficulty breathing, sudden severe weakness or numbness, severe bleeding), "
    "**STOP using this tool and call emergency services (911/112) or go to the nearest emergency room immediately!**"
)

# API & Patient Defaults Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_PATIENT_NAME = "John Doe"

# LLM Options
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2

# Form Selection Options
GENDER_OPTIONS = [
    "Female",
    "Male",
    "Non-binary",
    "Prefer not to say",
    "Other"
]

COMMON_SYMPTOMS = [
    "Fever",
    "Cough",
    "Shortness of breath",
    "Chest pain",
    "Headache",
    "Fatigue",
    "Runny nose",
    "Sore throat",
    "Nausea / Vomiting",
    "Abdominal pain",
    "Dizziness",
    "Joint pain",
    "Skin rash",
    "Loss of taste / smell",
    "Chills",
    "Muscle aches"
]

DURATION_OPTIONS = [
    "Less than 24 hours",
    "1 - 3 days",
    "4 - 7 days",
    "1 - 2 weeks",
    "More than 2 weeks"
]

LANGUAGE_OPTIONS = [
    "English",
    "Urdu",
    "Spanish",
    "French",
    "German",
    "Arabic",
    "Hindi"
]

CACHE_TYPES = [
    "Disabled",
    "InMemoryCache",
    "SQLiteCache"
]
