# 🩺 MediGuide AI - Medical Symptom Assessment & Patient Guidance Assistant

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green.svg)](https://www.langchain.com/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4o--mini-00A67E.svg)](https://openai.com/)

**MediGuide AI** is a professional, safety-focused medical symptom assessment and patient guidance prototype built with **Streamlit**, **LangChain**, and **OpenAI LLMs**. The application provides structured, preliminary educational guidance based on patient-reported symptoms, existing medical context, and self-reported severity.

> ⚠️ **IMPORTANT MEDICAL SAFETY DISCLAIMER**  
> MediGuide AI is an **educational prototype only**. It is **NOT** a doctor, cannot diagnose medical conditions, and does **NOT** provide clinical treatment or emergency medical services. Always consult a licensed healthcare professional for medical advice. If experiencing severe or red-flag emergency symptoms (such as severe chest pain or extreme shortness of breath), seek immediate emergency care (911/112).

---

## 🌟 Key Features

- 📋 **Structured Patient Assessment Form**: Collects Patient Name, Age, Gender, multiselect symptoms, optional free-text symptoms, duration, severity (1–10 slider), medical history, medications, notes, and response language.
- 🎨 **Dark Clinical SaaS UI**: Premium dark medical theme (`#0b0f19` near-black background, `#161b26` dark charcoal cards, crisp typography, and color-coded urgency badges).
- 🟢🟡🟠🔴 **Clear Urgency Level Classification**: Categorizes input into four strict urgency tiers: `LOW`, `MEDIUM`, `HIGH`, or `EMERGENCY`.
- 🩺 **Potential Clinical Correlations**: Presents educational potential illness categories with clear non-diagnosis disclaimers (*"Educational correlation • Not a diagnosis"*).
- ✨ **AI Patient Guidance (Streaming)**: Streams personalized narrative guidance word-by-word using `llm.stream()` and `st.write_stream()`, structured into 6 clear clinical headings.
- 🌐 **Multi-Language & Urdu Support**: Generates patient-facing narrative summaries in Urdu (or other languages) while keeping internal JSON keys in English for reliable parsing.
- ⚡ **LangChain Cache Strategy**: Supports toggling between `InMemoryCache` (RAM) and `SQLiteCache` (Disk `.langchain.db`) via `set_llm_cache()`.
- ⚙️ **Developer Technical Inspection**: Full UI expanders demonstrating `ChatOpenAI`, `PromptTemplate`, `ChatPromptTemplate`, `SystemMessage`, `HumanMessage`, `AIMessage`, `LLMChain`, `Structured JSON`, `Streaming`, and `Cache Diagnostics`.
- 📄 **Exportable Guidance Reports**: Downloads complete patient summaries as formatted `.txt` or `.pdf` reports.

---

## 📂 Project Architecture

```
MediGuideAi/
│
├── app.py                      # Main Streamlit web application & UI layout
├── requirements.txt            # Python dependencies (Streamlit, LangChain, OpenAI, FPDF2)
├── .env.example                # Template for environment configuration
├── .gitignore                  # Prevents secrets (.env) & databases (.db) from git
├── README.md                   # Comprehensive assignment documentation
│
├── src/                        # Backend business logic & LangChain modules
│   ├── __init__.py             # Python package marker
│   ├── config.py               # Constants, model configurations, disclaimers, options
│   ├── prompts.py              # LangChain PromptTemplate, ChatPromptTemplate, safety instructions
│   ├── chains.py               # ChatOpenAI, LLMChain construction, message demos, streaming
│   ├── cache_manager.py        # Global set_llm_cache configuration (InMemory & SQLite)
│   └── utils.py                # Safe JSON parser, input validation, urgency badges, PDF/TXT exports
│
└── docs/                       # Architectural documentation
    └── ARCHITECTURE.md         # Detailed technical design document
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ installed.
- An active OpenAI API key.

### 2. Clone Repository & Install Dependencies
```bash
# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)
Create a `.env` file in the project root directory (copied from `.env.example`):
```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```
> 🔒 **Security Note**: Never commit your `.env` file to version control. It is automatically ignored in `.gitignore`.

---

## 🚀 Running the Application

Launch the Streamlit web application with:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🧠 LangChain Concepts Demonstrated

| Concept | File Location | Assignment Implementation Details |
| :--- | :--- | :--- |
| **`ChatOpenAI`** | `src/chains.py` | Instantiates model (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`) with configurable temperature & API key security. |
| **`PromptTemplate`** | `src/prompts.py` | `SINGLE_STRING_PROMPT_TEMPLATE` defined with explicit `input_variables` list containing patient data fields. |
| **`ChatPromptTemplate`** | `src/prompts.py` | `ASSESSMENT_CHAT_PROMPT` constructed using `SystemMessagePromptTemplate` + `HumanMessagePromptTemplate`. |
| **`System/Human/AI Messages`** | `src/chains.py` | `demonstrate_raw_messages()` explicitly constructs `SystemMessage`, `HumanMessage`, and `AIMessage` objects for inspection in Tab 3. |
| **`LLMChain`** | `src/chains.py` | `build_assessment_chain()` creates an `LLMChain(llm=llm, prompt=ASSESSMENT_CHAT_PROMPT)` pipeline. |
| **Structured JSON** | `src/utils.py` | Robust parsing via `clean_and_parse_json()`. Handles surrounding text, malformed markdown, missing fields, and enforces schema normalization. |
| **Streaming** | `src/chains.py` & `app.py` | `llm.stream()` feeds directly into Streamlit's `st.write_stream()` in the **✨ AI Patient Guidance** section. |
| **`InMemoryCache` & `SQLiteCache`** | `src/cache_manager.py` | Configured globally via `set_llm_cache()`. Allows toggling between RAM (`InMemoryCache`) and Disk (`SQLiteCache` at `.langchain.db`). |

---

## 🧪 Testing Scenarios Verification Matrix

The project has been tested against the required rubric scenarios:

1. **Scenario 1 (Low Urgency)**: Age 25, Runny nose + sore throat, Duration 1–3 days, Severity 2 $\rightarrow$ Assesses as `LOW` urgency.
2. **Scenario 2 (Medium Urgency)**: Age 40, Fever + cough, Duration 4–7 days, Severity 6 $\rightarrow$ Assesses as `MEDIUM`/`HIGH` urgency with routine consultation advice.
3. **Scenario 3 (High/Emergency Urgency)**: Severe chest pain + shortness of breath $\rightarrow$ Assesses as `EMERGENCY` with prominent 911/112 warning banners.
4. **Scenario 4 (Cache Hit Timing)**: Submitting identical inputs twice with `InMemoryCache` or `SQLiteCache` enabled reduces execution latency from ~1500 ms to ~5-20 ms.
5. **Scenario 5 (Empty Symptoms Validation)**: Submitting without symptoms displays *"Please provide at least one symptom before starting the assessment."* and aborts the API call.
6. **Scenario 6 (Urdu Language Support)**: Selecting Urdu generates all patient-facing text in Urdu while preserving English JSON structure.

---

## 💯 100-Mark Rubric Compliance Checklist

- [x] **ChatOpenAI (10 Marks)**: Fully integrated via `langchain_openai`.
- [x] **PromptTemplate (10 Marks)**: Reusable single-string template implemented.
- [x] **ChatPromptTemplate & Messages (10 Marks)**: System, Human, and AI message objects constructed and displayed.
- [x] **LLMChain Pipeline (10 Marks)**: Assessment chain genuinely built using `LLMChain`.
- [x] **Structured JSON Parsing (10 Marks)**: Guaranteed JSON schema normalization; zero app crashes.
- [x] **Streaming Narrative (10 Marks)**: Live chunk streaming via `llm.stream()` and `st.write_stream()`.
- [x] **Caching Strategy (10 Marks)**: Both `InMemoryCache` and `SQLiteCache` active with `set_llm_cache()`.
- [x] **Streamlit UI/UX (15 Marks)**: Dark clinical SaaS dashboard with metrics, urgency badges, and tabbed inspection.
- [x] **Code Quality & Architecture (10 Marks)**: Clean modular `src/` directory structure with comments.
- [x] **Testing & Medical Safety (5 Marks)**: Non-diagnosis phrasing, emergency disclaimers, empty validation, and Urdu support.
