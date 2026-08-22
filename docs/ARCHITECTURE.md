# MediGuide AI - Technical Architecture & Design Document

## 1. Overview
MediGuide AI is designed as a modular, safety-conscious educational prototype that leverages Large Language Models (LLMs) to analyze patient-reported medical symptoms. The architecture strictly separates user interface logic (`app.py`) from backend orchestration, prompt engineering, caching, and data utility modules (`src/`).

---

## 2. Component Diagram

```
+-----------------------------------------------------------------------+
|                           Streamlit UI Layer                          |
|                                (app.py)                               |
|  +-------------------+  +--------------------+  +------------------+  |
|  | Patient Form Tab  |  | Guidance Dashboard |  | Tech Inspection  |  |
|  +---------+---------+  +---------+----------+  +--------+---------+  |
+------------|----------------------|----------------------|------------+
             |                      |                      |
             v                      v                      v
+-----------------------------------------------------------------------+
|                         Backend Logic Layer                           |
|                                (src/)                                 |
|                                                                       |
|  +------------------+     +-------------------+     +--------------+  |
|  |    config.py     |     |    prompts.py     |     |   utils.py   |  |
|  |  (App Options &  |     |  (ChatPrompts &   |     |  (JSON Parse |  |
|  |   Disclaimers)   |     |  System Safety)   |     |   & PDF/TXT) |  |
|  +------------------+     +---------+---------+     +--------------+  |
|                                     |                                 |
|                                     v                                 |
|                           +-------------------+                       |
|                           |     chains.py     |                       |
|                           | (LLMChain Pipeline|                       |
|                           |   & Streamer)     |                       |
|                           +---------+---------+                       |
|                                     |                                 |
|                                     v                                 |
|                           +-------------------+                       |
|                           | cache_manager.py  |                       |
|                           |  (set_llm_cache)  |                       |
|                           +-------------------+                       |
+-------------------------------------|---------------------------------+
                                      v
+-----------------------------------------------------------------------+
|                           LangChain & LLM                             |
|              (ChatOpenAI / InMemoryCache / SQLiteCache)               |
+-----------------------------------------------------------------------+
```

---

## 3. Data Pipeline & Logic Flow

1. **Input Validation**: `validate_symptom_inputs()` in `src/utils.py` checks if symptoms were supplied. If empty, the app displays a friendly error and aborts API invocation.
2. **Formatting**: `format_patient_inputs()` binds patient inputs (including Patient Name, Age, Gender, Symptoms, Duration, Severity, Medical History, Medications, Notes, and Language) into a normalized dictionary.
3. **Prompt Execution**: `run_assessment_chain()` in `src/chains.py` passes the dictionary into `LLMChain(llm=llm, prompt=ASSESSMENT_CHAT_PROMPT)`.
4. **Structured JSON Parsing**: `clean_and_parse_json()` strips markdown block fences, handles malformed JSON gracefully, and normalizes key values (`summary`, `possible_conditions`, `urgency_level`, `recommended_next_steps`, `questions_for_doctor`, `warning_signs`).
5. **Caching**: Global `set_llm_cache()` handles requests. Identical inputs return cached outputs instantly (<20 ms).
6. **Streaming Narrative**: `stream_narrative_guidance()` uses `llm.stream(NARRATIVE_CHAT_PROMPT)` and yields word chunks to `st.write_stream()` under 6 structured headings.
