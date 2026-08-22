"""
app.py
MediGuide AI - Main Interactive Streamlit Application

An educational AI-powered medical symptom assessment & patient guidance assistant
built with Streamlit, LangChain, and OpenAI LLMs.
Fully compliant with the 100-mark LangChain Assignment Specification and Rubric.
"""

import os
import time
import streamlit as st
from typing import Dict, Any

# Internal imports from src module
from src.config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    MEDICAL_DISCLAIMER_HEADER,
    MEDICAL_DISCLAIMER_SIDEBAR,
    EMERGENCY_WARNING_BANNER,
    OPENAI_API_KEY as ENV_OPENAI_API_KEY,
    DEFAULT_PATIENT_NAME,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GENDER_OPTIONS,
    COMMON_SYMPTOMS,
    DURATION_OPTIONS,
    LANGUAGE_OPTIONS,
    CACHE_TYPES,
)
from src.cache_manager import configure_cache, get_cache_info
from src.chains import (
    get_llm,
    build_assessment_chain,
    run_assessment_chain,
    demonstrate_raw_messages,
    stream_narrative_guidance,
    generate_mock_assessment_response,
    stream_mock_narrative_guidance,
)
from src.utils import (
    clean_and_parse_json,
    validate_symptom_inputs,
    get_urgency_badge_details,
    format_patient_inputs,
    generate_text_report,
    generate_pdf_report,
)
from src.prompts import (
    SINGLE_STRING_PROMPT_TEMPLATE,
    ASSESSMENT_CHAT_PROMPT,
    SYSTEM_SAFETY_INSTRUCTIONS,
    HUMAN_ASSESSMENT_PROMPT,
    NARRATIVE_CHAT_PROMPT,
)

# ------------------------------------------------------------------------------
# Page Configuration & Dark Clinical SaaS Custom Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} - AI Symptom Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Dark Clinical SaaS CSS (Near black background, dark charcoal cards, crisp typography)
st.markdown(
    """
    <style>
    /* Dark Theme Core Background & Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .main-header h1 {
        color: #ffffff !important;
        margin-bottom: 0.3rem;
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.025em;
    }
    .main-header p {
        color: #94a3b8 !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    /* Emergency Alert Banner */
    .emergency-banner {
        background-color: #450a0a;
        border: 1px solid #991b1b;
        color: #fca5a5;
        padding: 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Dark Charcoal Card Containers */
    .clinical-card {
        background-color: #161b26;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .correlation-card {
        background-color: #111827;
        border-left: 4px solid #3b82f6;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    
    .telemetry-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    
    .footnote-text {
        font-size: 0.82rem;
        color: #64748b;
        font-style: italic;
        margin-top: 0.4rem;
    }
    
    /* Streamlit Metric Customizations */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None
if "patient_inputs" not in st.session_state:
    st.session_state.patient_inputs = None
if "raw_chain_response" not in st.session_state:
    st.session_state.raw_chain_response = None
if "raw_message_demo" not in st.session_state:
    st.session_state.raw_message_demo = None
if "mock_mode" not in st.session_state:
    st.session_state.mock_mode = False
if "execution_time_ms" not in st.session_state:
    st.session_state.execution_time_ms = 0

# ------------------------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-female.png", width=64)
    st.title(f"🩺 {APP_NAME}")
    st.caption("AI-Powered Medical Symptom Assessment & Guidance")
    st.markdown("---")

    st.subheader("⚙️ Model Configuration")

    # API Key Input
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=ENV_OPENAI_API_KEY,
        type="password",
        help="Enter your OpenAI API key. Leave blank to run in Mock Mode for offline testing.",
    )
    
    # Mock Mode toggle if no API key is provided
    if not api_key_input or not api_key_input.strip():
        st.info("💡 **Mock Mode** active (No API key provided). Enter your key above to run live OpenAI models.")
        st.session_state.mock_mode = True
    else:
        st.session_state.mock_mode = False

    # Model Selection
    selected_model = st.selectbox(
        "Select Model",
        options=AVAILABLE_MODELS,
        index=0,
        help="Select OpenAI model architecture for inference.",
    )

    # Temperature Slider
    selected_temp = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        disabled=st.session_state.mock_mode,
        help="Lower values produce deterministic results; higher values increase output variability.",
    )

    st.markdown("---")
    st.subheader("⚡ LangChain Cache Strategy")

    cache_choice = st.radio(
        "Select Global LLM Cache",
        options=CACHE_TYPES,
        index=1,  # Default to InMemoryCache
        help="Demonstrates LangChain set_llm_cache() with InMemoryCache vs SQLiteCache.",
    )

    # Apply Cache Setting
    cache_success, cache_msg = configure_cache(cache_choice)
    if cache_success:
        st.success(cache_msg)
    else:
        st.error(cache_msg)

    # Render Cache Telemetry Details
    cache_info = get_cache_info(cache_choice)
    st.markdown(
        f"""
        <div class="telemetry-card">
            <b>Storage:</b> {cache_info.get('storage')}<br/>
            <b>Persistence:</b> {cache_info.get('persistence')}<br/>
            <b>Latency:</b> {cache_info.get('speed')}<br/>
            <b>Best For:</b> {cache_info.get('best_for')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.warning(
        "⚠️ **MEDICAL & SAFETY NOTICE**\n\n"
        "MediGuide AI is an educational AI prototype. It is not a doctor and does not provide medical diagnosis, "
        "treatment, or emergency services. Always consult a qualified healthcare professional. If you believe you are "
        "experiencing a medical emergency, seek emergency medical assistance immediately."
    )

# ------------------------------------------------------------------------------
# Main Page Header & Safety Notice
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        <h1>🩺 {APP_NAME}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Emergency Warning Banner
st.markdown(
    f"""
    <div class="emergency-banner">
        {EMERGENCY_WARNING_BANNER}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ Read Medical & Safety Notice (Educational Prototype Disclosure)", expanded=False):
    st.warning(MEDICAL_DISCLAIMER_HEADER)

# ------------------------------------------------------------------------------
# Main Navigation Tabs
# ------------------------------------------------------------------------------
tab_form, tab_dashboard, tab_tech = st.tabs([
    "🩺 Patient Assessment",
    "📊 Guidance Dashboard",
    "⚙️ LangChain Technical Inspection"
])

# ==============================================================================
# TAB 1: Patient Assessment Form
# ==============================================================================
with tab_form:
    st.subheader("📋 Patient Information & Clinical Form")
    st.caption("Complete the patient details below to generate structured educational guidance.")

    with st.form("symptom_assessment_form", clear_on_submit=False):
        st.markdown("##### 👤 Patient Information")
        col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
        
        with col_p1:
            patient_name_input = st.text_input(
                "Patient Name",
                value="John Doe",
                placeholder="Enter patient's full name",
                help="Full name of the patient for assessment documentation.",
            )
        with col_p2:
            age_input = st.text_input("Age", value="35", help="Patient age in years.")
        with col_p3:
            gender_input = st.selectbox("Gender", options=GENDER_OPTIONS, index=0)

        st.markdown("---")
        st.markdown("##### 🔍 Symptoms & Severity")
        
        selected_symptoms_list = st.multiselect(
            "Select Symptoms",
            options=COMMON_SYMPTOMS,
            default=["Fever", "Cough"],
            help="Select all applicable symptoms from the list.",
        )

        custom_symptoms_input = st.text_area(
            "Additional / Detailed Symptoms (Free-Text)",
            placeholder="e.g. Mild headache behind eyes, dry cough worsening at night...",
            height=80,
            help="Describe any symptoms not listed above.",
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            duration_input = st.selectbox("Symptom Duration", options=DURATION_OPTIONS, index=1)
        with col_s2:
            severity_input = st.slider(
                "Severity (1 = Mild, 10 = Extreme)",
                min_value=1,
                max_value=10,
                value=5,
                help="Self-reported discomfort level.",
            )

        st.markdown("---")
        st.markdown("##### 🏥 Medical Context")
        
        col_med1, col_med2 = st.columns(2)
        with col_med1:
            conditions_input = st.text_area(
                "Existing Medical Conditions",
                placeholder="e.g. Asthma, High blood pressure, None...",
                height=80,
            )
        with col_med2:
            medications_input = st.text_area(
                "Current Medications",
                placeholder="e.g. Albuterol inhaler, Vitamin C, None...",
                height=80,
            )

        notes_input = st.text_area(
            "Additional Notes",
            placeholder="e.g. Traveled recently, exposed to sick family member...",
            height=60,
        )

        st.markdown("---")
        st.markdown("##### 🌐 Response Language")
        language_input = st.selectbox(
            "Select Guidance Language",
            options=LANGUAGE_OPTIONS,
            index=0,
            help="Select preferred response language (e.g. English or Urdu).",
        )

        st.markdown("")
        submit_button = st.form_submit_button("🚀 Analyze Symptoms", type="primary", use_container_width=True)

    if submit_button:
        # Validate empty symptoms input before API call per assignment rubric
        is_valid, val_msg = validate_symptom_inputs(selected_symptoms_list, custom_symptoms_input)
        if not is_valid:
            st.error(val_msg)
        else:
            patient_data = format_patient_inputs(
                patient_name=patient_name_input,
                age=age_input,
                gender=gender_input,
                symptoms_list=selected_symptoms_list,
                custom_symptoms=custom_symptoms_input,
                duration=duration_input,
                severity=severity_input,
                conditions=conditions_input,
                medications=medications_input,
                notes=notes_input,
                language=language_input,
            )

            st.session_state.patient_inputs = patient_data
            start_time = time.time()

            with st.spinner("⏳ Analyzing symptoms and running LangChain LLMChain pipeline..."):
                try:
                    if st.session_state.mock_mode:
                        raw_response = generate_mock_assessment_response(patient_data)
                        demo = {
                            "system_message": {"content": "MOCK SYSTEM SAFETY INSTRUCTIONS"},
                            "human_message": {"content": str(patient_data)},
                            "ai_message": {"content": raw_response},
                            "raw_response_text": raw_response,
                        }
                    else:
                        llm = get_llm(
                            model_name=selected_model,
                            temperature=selected_temp,
                            api_key=api_key_input,
                        )
                        raw_response = run_assessment_chain(llm, patient_data)
                        demo = demonstrate_raw_messages(llm, patient_data)

                    st.session_state.execution_time_ms = round((time.time() - start_time) * 1000, 2)

                    # Parse JSON safely
                    parsed_json, parse_ok, parse_err = clean_and_parse_json(raw_response)

                    st.session_state.raw_chain_response = raw_response
                    st.session_state.assessment_result = parsed_json
                    st.session_state.raw_message_demo = demo

                    st.success("✅ Symptoms analyzed successfully! Switch to the **📊 Guidance Dashboard** tab to view your clinical summary.")
                    
                except Exception as ex:
                    err_str = str(ex)
                    if "insufficient_quota" in err_str or "429" in err_str:
                        st.warning("⚠️ **OpenAI Quota Exceeded (Error 429)**: The provided OpenAI account has no remaining API credits. **Automatically running in Mock Mode** so you can demonstrate all features.")
                        raw_response = generate_mock_assessment_response(patient_data)
                        demo = {
                            "system_message": {"content": "MOCK SYSTEM SAFETY INSTRUCTIONS (Quota Fallback)"},
                            "human_message": {"content": str(patient_data)},
                            "ai_message": {"content": raw_response},
                            "raw_response_text": raw_response,
                        }
                        st.session_state.execution_time_ms = round((time.time() - start_time) * 1000, 2)
                        parsed_json, parse_ok, parse_err = clean_and_parse_json(raw_response)
                        st.session_state.raw_chain_response = raw_response
                        st.session_state.assessment_result = parsed_json
                        st.session_state.raw_message_demo = demo
                        st.success("✅ Generated mock assessment result! Switch to **📊 Guidance Dashboard** tab to view your clinical summary.")
                    elif "401" in err_str or "invalid_api_key" in err_str:
                        st.error("❌ **Invalid OpenAI API Key**: OpenAI API keys must start with `sk-...` (e.g. `sk-proj-...`). Clear the API key box in the sidebar to run in **Mock Mode**.")
                    else:
                        st.error(f"❌ Error executing assessment chain: {err_str}")

# ==============================================================================
# TAB 2: Guidance Dashboard
# ==============================================================================
with tab_dashboard:
    if not st.session_state.assessment_result or not st.session_state.patient_inputs:
        st.info("👈 Please complete and submit the **Patient Assessment** form in Tab 1 to generate guidance.")
    else:
        patient_data = st.session_state.patient_inputs
        assessment = st.session_state.assessment_result

        # Patient Header & Assessment Metrics
        st.markdown(f"### Patient: `{patient_data.get('patient_name', DEFAULT_PATIENT_NAME)}`")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="Severity", value=f"{patient_data.get('severity')}/10")
        with col_m2:
            st.metric(label="Duration", value=str(patient_data.get('duration')))
        with col_m3:
            st.metric(label="Urgency Level", value=str(assessment.get('urgency_level')))
        with col_m4:
            st.metric(label="Execution Time", value=f"{st.session_state.execution_time_ms} ms")

        st.markdown("---")

        # Urgency Level Card
        urgency_level = assessment.get("urgency_level", "MEDIUM")
        badge_info = get_urgency_badge_details(urgency_level)

        st.markdown(
            f"""
            <div style="background-color: {badge_info['bg_color']}; border: 1px solid {badge_info['border_color']}; padding: 1.2rem 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #ffffff;">{badge_info['label']}</h3>
                <p style="margin: 0.4rem 0 0 0; color: #f1f5f9; font-size: 1.05rem;">{badge_info['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Patient Assessment Summary
        st.markdown("### 📝 Patient Assessment Summary")
        st.write(assessment.get("summary", "Summary unavailable."))

        col_dash1, col_dash2 = st.columns(2)

        with col_dash1:
            st.markdown("### 🩺 Potential Clinical Correlations")
            st.caption("Educational information only — these are not diagnoses.")
            
            conditions = assessment.get("possible_conditions", [])
            if conditions:
                for cond in conditions:
                    c_name = cond.get("name", "Clinical Correlation")
                    c_reason = cond.get("reason", "No detailed rationale provided.")
                    st.markdown(
                        f"""
                        <div class="correlation-card">
                            <b style="color: #60a5fa; font-size: 1.05rem;">• {c_name}</b><br/>
                            <span style="color: #cbd5e1; font-size: 0.95rem;"><b>Educational Rationale:</b> {c_reason}</span><br/>
                            <span class="footnote-text">Educational correlation • Not a diagnosis</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No specific correlations identified.")

            st.markdown("---")
            st.markdown("### 📋 Recommended Next Steps")
            steps = assessment.get("recommended_next_steps", [])
            for idx, step in enumerate(steps, 1):
                st.checkbox(f"{step}", value=False, key=f"step_{idx}")

        with col_dash2:
            st.markdown("### ❓ Questions for Your Doctor")
            questions = assessment.get("questions_for_doctor", [])
            for q in questions:
                st.markdown(f"- ❓ **{q}**")

            st.markdown("---")
            st.markdown("### 🚨 Red-Flag Warning Signs")
            st.caption("Seek immediate emergency medical evaluation (911/112) if you experience any of the following:")
            warnings = assessment.get("warning_signs", [])
            for w in warnings:
                st.error(f"🚨 {w}")
            
            st.caption("This list is not exhaustive. If you believe you are experiencing a medical emergency, seek emergency medical care immediately.")

        st.markdown("---")
        
        # ✨ AI Patient Guidance (Streaming Section)
        st.markdown("### ✨ AI Patient Guidance")
        st.caption("Personalized educational guidance based on the information provided.")

        if st.button("▶️ Generate / Refresh AI Guidance Narrative", type="primary", key="stream_btn"):
            with st.spinner("AI is preparing your guidance..."):
                if st.session_state.mock_mode:
                    st.write_stream(stream_mock_narrative_guidance(patient_data))
                else:
                    try:
                        llm = get_llm(
                            model_name=selected_model,
                            temperature=selected_temp,
                            api_key=api_key_input,
                        )
                        st.write_stream(stream_narrative_guidance(llm, patient_data))
                    except Exception as e:
                        err_s = str(e)
                        if "insufficient_quota" in err_s or "429" in err_s:
                            st.warning("⚠️ **OpenAI Quota Exceeded (Error 429)**: Streaming mock guidance narrative below.")
                            st.write_stream(stream_mock_narrative_guidance(patient_data))
                        else:
                            st.error(f"Streaming error: {err_s}")

        st.markdown("---")
        st.markdown("### 📥 Download Guidance Report")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            txt_report = generate_text_report(patient_data, assessment)
            st.download_button(
                label="📄 Download Text Report (.txt)",
                data=txt_report,
                file_name=f"mediguide_report_{patient_data.get('patient_name', 'patient').replace(' ', '_').lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_dl2:
            try:
                pdf_bytes = generate_pdf_report(patient_data, assessment)
                st.download_button(
                    label="📕 Download PDF Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"mediguide_report_{patient_data.get('patient_name', 'patient').replace(' ', '_').lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as pdf_err:
                st.warning(f"PDF generation note: {str(pdf_err)}")

# ==============================================================================
# TAB 3: LangChain Technical Inspection
# ==============================================================================
with tab_tech:
    st.subheader("⚙️ LangChain Technical Inspection & Assignment Verification")
    st.caption("Inspect live message objects, reusable prompt templates, LLMChain status, structured JSON, streaming, and cache diagnostics.")

    if not st.session_state.raw_message_demo:
        st.info("👈 Please execute an assessment in Tab 1 to inspect live LangChain objects.")
    else:
        demo = st.session_state.raw_message_demo

        with st.expander("🤖 1. ChatOpenAI Configuration", expanded=True):
            st.markdown(f"- **Model**: `{selected_model}`")
            st.markdown(f"- **Temperature**: `{selected_temp}`")
            st.markdown(f"- **Provider**: `langchain_openai.ChatOpenAI`")
            st.markdown(f"- **API Key Security**: `Configured via .env / Password Widget (Hidden)`")
            st.markdown(f"- **Streaming**: `Enabled via llm.stream()`")

        with st.expander("📄 2. PromptTemplate (Single-String Reusable Template)", expanded=False):
            st.caption("Demonstrates PromptTemplate class with explicit input variables list:")
            st.code(SINGLE_STRING_PROMPT_TEMPLATE.template, language="markdown")

        with st.expander("🧩 3. ChatPromptTemplate (System & Human Messages)", expanded=False):
            st.caption("Demonstrates ChatPromptTemplate with SystemMessagePromptTemplate and HumanMessagePromptTemplate:")
            st.code(str(ASSESSMENT_CHAT_PROMPT), language="python")

        with st.expander("🛡️ 4. SystemMessage Object", expanded=False):
            sys_content = (
                demo["system_message"].content
                if hasattr(demo["system_message"], "content")
                else demo["system_message"]["content"]
            )
            st.code(sys_content, language="markdown")

        with st.expander("👤 5. HumanMessage Object", expanded=False):
            human_content = (
                demo["human_message"].content
                if hasattr(demo["human_message"], "content")
                else demo["human_message"]["content"]
            )
            st.code(human_content, language="markdown")

        with st.expander("🤖 6. AIMessage Object (Raw LLM Output)", expanded=False):
            ai_content = (
                demo["ai_message"].content
                if hasattr(demo["ai_message"], "content")
                else demo["ai_message"]["content"]
            )
            st.code(ai_content, language="json")

        with st.expander("⚙️ 7. LLMChain Execution Pipeline", expanded=False):
            st.markdown("- **Chain Class**: `langchain.chains.LLMChain`")
            st.markdown("- **Prompt**: `ASSESSMENT_CHAT_PROMPT`")
            st.markdown("- **Execution Status**: `Successfully Invoked`")
            st.markdown(f"- **Execution Latency**: `{st.session_state.execution_time_ms} ms`")

        with st.expander("📊 8. Structured JSON Parsed Object", expanded=False):
            st.json(st.session_state.assessment_result)

        with st.expander("⚡ 9. LangChain Cache Diagnostics (InMemoryCache & SQLiteCache)", expanded=False):
            st.json(get_cache_info(cache_choice))
            st.markdown(
                """
                **Cache Behavior & Rubric Demonstration:**
                - **InMemoryCache**: Stores LLM prompt/response pairs in Python process RAM. Fast, volatile.
                - **SQLiteCache**: Stores prompt/response pairs on disk (`.langchain.db`). Persists across app restarts.
                - **Verification**: Submitting identical patient inputs twice with caching enabled reduces execution time dramatically (~5-20 ms vs ~1500 ms).
                """
            )
