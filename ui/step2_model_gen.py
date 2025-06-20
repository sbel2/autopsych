# File: ui/step2_model_gen.py

import streamlit as st
import pandas as pd
from config import GENERAL_PROMPT, GROQ_API_KEY, GROQ_API_URL
from llm_integration import call_groq_api

def render_step2():
    """Renders the UI for Step 2: Model Generation."""
    st.header("Step 2: Generate a Predictive Model with an LLM")
    st.info("Review the prompt below, edit it if needed, and then generate a model with the LLM.")

    prompt_template = GENERAL_PROMPT
    data_summary_text = st.session_state.get('editable_data', pd.DataFrame()).head(10).to_string()
    prompt_text = prompt_template.format(data_summary=data_summary_text)

    st.subheader("Editable LLM Prompt")
    st.session_state.llm_prompt = st.text_area("LLM Prompt:", value=prompt_text, height=400)

    if st.button("Generate Python Model via API", type="primary"):
        with st.spinner("Calling Groq API... Please wait for the live result."):
            generated_code = call_groq_api(st.session_state.llm_prompt, GROQ_API_KEY, GROQ_API_URL)
            if generated_code:
                st.session_state.model_code = generated_code
            else:
                st.error("Model generation failed. Please check the error messages in the expanders above and try again.")

    st.subheader("Editable Python Model")
    st.info("The LLM might generate code with minor syntax errors. Please fix them before using the model.")
    st.session_state.model_code = st.text_area("Python Model Code:", value=st.session_state.model_code, height=350, key="model_editor",
                             placeholder="Click 'Generate Python Model' above to have the LLM create code here.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Go Back to Data"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.session_state.model_code:
            if st.button("Use this Model for Testing →", type="primary"):
                st.session_state.step = 3
                st.rerun()