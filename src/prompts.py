"""
prompts.py
LangChain Prompt Templates & JSON Schema Instructions for MediGuide AI

Defines PromptTemplate, ChatPromptTemplate, and safety system instructions
for structured JSON output generation and narrative streaming.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Mandatory JSON Schema Definition required by assignment specification
ASSESSMENT_JSON_SCHEMA = """{{
  "summary": "A concise, clinical educational summary of the reported symptoms and patient profile.",
  "possible_conditions": [
    {{
      "name": "Condition or Illness Category (e.g. Upper Respiratory Tract Illness)",
      "reason": "Educational explanation of why these symptoms may be correlated. Never state a confirmed diagnosis."
    }}
  ],
  "urgency_level": "LOW | MEDIUM | HIGH | EMERGENCY",
  "recommended_next_steps": [
    "Actionable, supportive step 1...",
    "Actionable step 2..."
  ],
  "questions_for_doctor": [
    "Question 1 to ask healthcare provider...",
    "Question 2..."
  ],
  "warning_signs": [
    "Red-flag warning sign 1 requiring immediate medical evaluation...",
    "Warning sign 2..."
  ]
}}"""

SYSTEM_SAFETY_INSTRUCTIONS = """You are MediGuide AI, an intelligent, empathetic, and safety-focused preliminary medical symptom assessment assistant.

CRITICAL SAFETY & MEDICAL RULES:
1. You are an EDUCATIONAL PROTOTYPE ONLY. You are NOT a doctor and CANNOT diagnose medical conditions.
2. NEVER present any output as a confirmed diagnosis (e.g. NEVER say "You have X disease"). Always use phrasing such as "may be associated with", "possible correlation", or "could be consistent with".
3. Every response MUST categorize the situation into one of four EXACT urgency levels:
   - "LOW": Mild, self-limiting symptoms suitable for routine self-monitoring.
   - "MEDIUM": Moderate symptoms requiring non-urgent consultation with a healthcare professional.
   - "HIGH": Severe or worsening symptoms needing prompt evaluation at an urgent care center or doctor office.
   - "EMERGENCY": Red-flag symptoms (e.g., severe chest pain, extreme shortness of breath, sudden neurological deficits, major bleeding) requiring IMMEDIATE emergency care (911/112 or emergency room).
4. URDU & MULTI-LANGUAGE SUPPORT:
   - If requested response language is Urdu, generate all text values inside the JSON (summary, condition names, reasons, steps, questions, warning signs) in URDU language using Urdu script.
   - The JSON structure and property keys (e.g. "summary", "possible_conditions", "urgency_level", etc.) MUST remain in English so parsing succeeds.
5. You MUST return ONLY a valid JSON object matching the requested schema exactly. Do NOT add preamble or markdown code fences outside the JSON.
"""

HUMAN_ASSESSMENT_PROMPT = """Analyze the following patient assessment input and provide structured preliminary guidance:

Patient Profile & Clinical Inputs:
- Patient Name: {patient_name}
- Age: {age}
- Gender: {gender}
- Primary & Secondary Symptoms: {symptoms}
- Symptom Duration: {duration}
- Self-Reported Severity (1-10): {severity}/10
- Existing Medical Conditions: {existing_conditions}
- Current Medications: {medications}
- Additional Patient Notes: {additional_notes}
- Requested Response Language: {language}

REQUIRED OUTPUT FORMAT:
Return ONLY a valid JSON object strictly matching this schema format:
""" + ASSESSMENT_JSON_SCHEMA

# 1. ChatPromptTemplate implementation (System + Human role messages)
ASSESSMENT_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_SAFETY_INSTRUCTIONS),
    HumanMessagePromptTemplate.from_template(HUMAN_ASSESSMENT_PROMPT)
])

# 2. Reusable PromptTemplate implementation (Single-string template with variables demonstration)
SINGLE_STRING_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "patient_name", "age", "gender", "symptoms", "duration", "severity",
        "existing_conditions", "medications", "additional_notes", "language"
    ],
    template=SYSTEM_SAFETY_INSTRUCTIONS + "\n\n" + HUMAN_ASSESSMENT_PROMPT
)

# 3. Narrative Streaming Prompt Templates
NARRATIVE_SYSTEM_PROMPT = """You are MediGuide AI, an empathetic medical assessment assistant.
Provide a structured, easy-to-read narrative guidance summary for the patient based on their details.
State clearly that this is educational advice only and guide them on next steps.
Language of response MUST be: {language}."""

NARRATIVE_HUMAN_PROMPT = """Patient Details:
- Patient Name: {patient_name}
- Age: {age}, Gender: {gender}
- Symptoms: {symptoms} (Duration: {duration}, Severity: {severity}/10)
- Existing Conditions: {existing_conditions}
- Current Medications: {medications}
- Notes: {additional_notes}
- Language: {language}

Provide a personalized, structured narrative formatted into these EXACT 6 markdown headings (in {language}):

### Your Current Assessment
Briefly summarize the reported symptoms, severity, and urgency level for {patient_name}.

### What These Symptoms May Indicate
Explain potential educational correlations without making a diagnosis (use phrases like "may be associated with").

### What You Can Do Now
Provide safe, supportive guidance and self-care steps.

### What to Monitor
Explain specific changes or progression the patient should watch closely.

### When to Seek Medical Care
Explain when to contact a doctor and when immediate emergency care (911/112) is required.

### Important Safety Notice
Remind the patient that MediGuide AI is an educational prototype and not a replacement for a clinical diagnosis.
"""

NARRATIVE_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(NARRATIVE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(NARRATIVE_HUMAN_PROMPT)
])
