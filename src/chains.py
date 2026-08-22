"""
chains.py
LangChain Chain Execution & LLM Integration for MediGuide AI

Implements ChatOpenAI model instantiation, LLMChain construction,
explicit System/Human/AI message demonstrations, and live narrative streaming.
"""

from typing import Dict, Any, Generator
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Legacy & Runnable chain imports for compatibility
try:
    from langchain.chains import LLMChain
except ImportError:
    LLMChain = None

from src.prompts import (
    ASSESSMENT_CHAT_PROMPT,
    SYSTEM_SAFETY_INSTRUCTIONS,
    HUMAN_ASSESSMENT_PROMPT,
    NARRATIVE_CHAT_PROMPT
)


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2, api_key: str = "") -> ChatOpenAI:
    """
    Initializes and returns a ChatOpenAI model instance.
    """
    if not api_key:
        raise ValueError("OpenAI API Key is missing. Please provide a valid key in .env or the sidebar.")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key
    )


def build_assessment_chain(llm: ChatOpenAI):
    """
    Builds and returns a reusable assessment chain using LLMChain.
    Demonstrates LLMChain usage as required by the assignment specification.
    """
    if LLMChain is not None:
        return LLMChain(llm=llm, prompt=ASSESSMENT_CHAT_PROMPT, verbose=True)
    else:
        # Runnable fallback if LLMChain module path is updated in environment
        return ASSESSMENT_CHAT_PROMPT | llm


def run_assessment_chain(llm: ChatOpenAI, inputs: Dict[str, Any]) -> str:
    """
    Executes the assessment chain with formatted patient inputs and returns raw string response.
    """
    chain = build_assessment_chain(llm)
    
    if hasattr(chain, "run"):
        # LLMChain invocation
        return chain.run(**inputs)
    elif hasattr(chain, "invoke"):
        # Runnable composition invocation
        response = chain.invoke(inputs)
        return response.content if hasattr(response, "content") else str(response)
    else:
        raise RuntimeError("Failed to execute assessment chain.")


def demonstrate_raw_messages(llm: ChatOpenAI, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Demonstrates explicit construction and inspection of SystemMessage, HumanMessage, and AIMessage objects
    as mandated by section 7 of the assignment spec.
    """
    formatted_system = SYSTEM_SAFETY_INSTRUCTIONS.format(language=inputs.get("language", "English"))
    formatted_human = HUMAN_ASSESSMENT_PROMPT.format(**inputs)

    sys_msg = SystemMessage(content=formatted_system)
    human_msg = HumanMessage(content=formatted_human)

    # Invoke model with raw message list
    ai_response = llm.invoke([sys_msg, human_msg])

    ai_msg = AIMessage(content=ai_response.content)

    return {
        "system_message": sys_msg,
        "human_message": human_msg,
        "ai_message": ai_msg,
        "raw_response_text": ai_response.content
    }


def stream_narrative_guidance(llm: ChatOpenAI, inputs: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Streams final patient guidance narrative chunk-by-chunk using llm.stream()
    for live rendering via Streamlit's st.write_stream().
    """
    messages = NARRATIVE_CHAT_PROMPT.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def generate_mock_assessment_response(inputs: Dict[str, Any]) -> str:
    """
    Generates a realistic mock JSON assessment response for offline testing or demonstration
    when an OpenAI API key is unavailable.
    """
    patient_name = inputs.get("patient_name", "John Doe")
    symptoms = str(inputs.get("symptoms", "")).lower()
    severity = int(inputs.get("severity", 5))
    language = inputs.get("language", "English")

    urgency = "MEDIUM"
    if "chest pain" in symptoms or "shortness of breath" in symptoms or severity >= 8:
        urgency = "EMERGENCY" if "chest pain" in symptoms and severity >= 8 else "HIGH"
    elif severity <= 3 and ("runny nose" in symptoms or "fatigue" in symptoms):
        urgency = "LOW"

    if language == "Urdu":
        return f"""{{
  "summary": "مریض {patient_name} نے علامات ({inputs.get('symptoms')}) کی اطلاع دی ہے جو {inputs.get('duration')} سے جاری ہیں۔ شدت کی درجہ بندی {inputs.get('severity')}/10 ہے۔",
  "possible_conditions": [
    {{
      "name": "عام وائرل انفیکشن / نزلہ زکام",
      "reason": "یہ علامات عام سانس کی نالی کے انفیکشن سے مطابقت رکھتی ہیں۔"
    }},
    {{
      "name": "ماحولیاتی یا مادی ردعمل",
      "reason": "موجودہ شدت اور دورانیے سے مطابقت رکھتا ہے۔"
    }}
  ],
  "urgency_level": "{urgency}",
  "recommended_next_steps": [
    "اگلے 24 سے 48 گھنٹوں میں مناسب آرام کریں اور پانی کا استعمال زیادہ رکھیں۔",
    "بخار اور علامات کی باقاعدگی سے نگرانی کریں۔",
    "اگر علامات برقرار رہیں یا بگڑ جائیں تو ڈاکٹر سے مشورہ کریں۔"
  ],
  "questions_for_doctor": [
    "کیا یہ علامات میری سابقہ طبی حالت یا ادویات سے متعلق ہو سکتی ہیں؟",
    "کن ہنگامی علامات کی صورت میں فوری طور پر ہسپتال جانا چاہیے؟"
  ],
  "warning_signs": [
    "سانس لینے میں شدید دشواری یا سینے میں مسلسل درد",
    "تیز بخار جو عام ادویات سے کم نہ ہو",
    "شدید کمزوری، چکر آنا یا بے ہوشی"
  ]
}}"""

    return f"""{{
  "summary": "Patient {patient_name} reports {inputs.get('symptoms', 'unspecified symptoms')} lasting for {inputs.get('duration', 'a few days')} with a self-reported severity of {inputs.get('severity', '5')}/10.",
  "possible_conditions": [
    {{
      "name": "Upper Respiratory Illness / General Viral Infection",
      "reason": "This symptom pattern may be associated with common respiratory illnesses. This is not a diagnosis."
    }},
    {{
      "name": "Symptomatic Response to Environmental or Mild Strain Factors",
      "reason": "Correlates with reported duration ({inputs.get('duration', 'N/A')}) and severity rating."
    }}
  ],
  "urgency_level": "{urgency}",
  "recommended_next_steps": [
    "Ensure adequate hydration and get sufficient rest over the next 24-48 hours.",
    "Monitor temperature and symptom progression regularly.",
    "Schedule a routine consultation with a healthcare professional if symptoms persist or worsen."
  ],
  "questions_for_doctor": [
    "Could these symptoms be related to my existing medical conditions or current medications?",
    "What specific warning signs should prompt immediate emergency care?",
    "Are any diagnostic tests appropriate for these symptoms?"
  ],
  "warning_signs": [
    "Sudden onset of severe shortness of breath or difficulty breathing",
    "Persistent chest pain, pressure, or tightness",
    "High fever unresponsive to standard over-the-counter fever reducers",
    "Sudden severe weakness, confusion, or dizziness"
  ]
}}"""


def stream_mock_narrative_guidance(inputs: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Streams a mock narrative response structured into the 6 required sections word-by-word for testing st.write_stream().
    """
    patient_name = inputs.get("patient_name", "John Doe")
    language = inputs.get("language", "English")
    symptoms = inputs.get("symptoms", "symptoms")
    severity = inputs.get("severity", 5)

    if language == "Urdu":
        text = f"""### Your Current Assessment
مریض **{patient_name}** کی فراہم کردہ معلومات کے مطابق علامات: {symptoms}، دورانیہ: {inputs.get('duration')}، اور شدت: {severity}/10 ہے۔

### What These Symptoms May Indicate
یہ علامات عام سانس کی بیماریوں یا وائرل انفیکشن سے متعلق ہو سکتی ہیں۔ یہ کوئی حتمی تشخيص نہیں ہے۔

### What You Can Do Now
مناسب آرام کریں، وافر مقدار میں پانی پیئیں، اور بخار کی باقاعدگی سے نگرانی کریں۔

### What to Monitor
اگر آپ کو سانس لینے میں دشواری، سینے میں درد، یا تیز بخار محسوس ہو تو فوراً توجہ دیں۔

### When to Seek Medical Care
اگر علامات 48 گھنٹوں سے زیادہ برقرار رہیں تو ڈاکٹر سے رجوع کریں۔

### Important Safety Notice
MediGuide AI صرف ایک تعلیمی پروٹوائپ ہے اور ڈاکٹر کا متبادل نہیں ہے۔"""
    else:
        text = f"""### Your Current Assessment
Patient **{patient_name}** has reported symptoms including {symptoms} with a duration of {inputs.get('duration')} and self-reported severity of {severity}/10.

### What These Symptoms May Indicate
These symptoms may be associated with common upper respiratory illnesses or general viral strain. This is an educational correlation and not a confirmed diagnosis.

### What You Can Do Now
Ensure adequate rest, maintain good hydration, and monitor your body temperature regularly.

### What to Monitor
Watch for any progression in symptom severity, development of high fever, or onset of chest discomfort.

### When to Seek Medical Care
Schedule a consultation with a qualified healthcare provider if symptoms persist beyond a few days or worsen significantly. For emergency symptoms (e.g. severe chest pain or shortness of breath), seek emergency medical care immediately.

### Important Safety Notice
MediGuide AI is an educational AI prototype designed for preliminary guidance only. It is not a doctor and does not provide clinical diagnoses or treatments."""

    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.02)
