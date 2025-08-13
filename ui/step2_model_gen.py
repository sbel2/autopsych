# File: ui/step2_model_gen.py

import streamlit as st
import pandas as pd
from config import GENERAL_PROMPT, GOOGLE_API_KEY, GEMINI_MODEL
from llm_integration import call_gemini_api

def render_step2():
    """Renders the UI for Step 2: Model Generation."""
    st.header("Step 2: Generate a Predictive Model with an LLM")
    st.info("Review the prompt below, edit it if needed, and then generate a model with the LLM.")

    # Get the data
    df = st.session_state.get('editable_data', pd.DataFrame())
    
    # Create the full data text for the prompt
    full_data_text = '\n'.join([
        '| ' + ' | '.join(df.columns) + ' |',
        '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
    ])
    
    for _, row in df.iterrows():
        row_values = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                val = ''
            row_values.append(str(val))
        full_data_text += '\n| ' + ' | '.join(row_values) + ' |'
    
    # Just show a simple message about the data being included
    st.info(f"Data loaded: {len(df)} rows")
    
    prompt_text = GENERAL_PROMPT.replace('{{data_summary}}', full_data_text)

    st.subheader("Editable LLM Prompt")
    st.session_state.llm_prompt = st.text_area("LLM Prompt:", value=prompt_text, height=400)

    if st.button("Generate Python Model via API", type="primary"):
        with st.spinner("Calling Google Gemini API... Please wait for the live result."):
            generated_code = call_gemini_api(st.session_state.llm_prompt, GOOGLE_API_KEY, GEMINI_MODEL)
            if generated_code:
                st.session_state.model_code = generated_code
            else:
                st.error("Model generation failed. Please check the error messages in the expanders above and try again.")

    st.subheader("Editable Python Model")
    st.info("The LLM might generate code with minor syntax errors. Please fix them before using the model.")
    
    # Load best model content if available
    try:
        with open('best_model', 'r') as f:
            best_model_content = f.read()
        
        # If no model code in session state yet, use the best model
        if 'model_code' not in st.session_state or not st.session_state.model_code:
            st.session_state.model_code = best_model_content
    except FileNotFoundError:
        best_model_content = ""
    
    # Text area for editing the model code
    st.session_state.model_code = st.text_area(
        "Python Model Code:", 
        value=st.session_state.model_code, 
        height=350, 
        key="model_editor",
        placeholder="Enter your Python model code here or click 'Generate Python Model' above."
    )
    
    # Add a button to load the best model
    if best_model_content:
        if st.button("🔄 Load Best Model"):
            st.session_state.model_code = best_model_content
            st.rerun()
    
    # Navigation buttons at the bottom
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Go Back to Data"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Use this Model for Testing →", type="primary"):
            try:
                # Validate the code when proceeding
                local_namespace = {}
                exec(st.session_state.model_code, globals(), local_namespace)
                if 'predict_choice_proportions' not in local_namespace:
                    st.error("❌ The code must define a function named 'predict_choice_proportions'")
                else:
                    st.session_state.step = 3
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error in model code: {str(e)}")
    
    # Show a small info message about the required function
    st.caption("💡 Make sure your code defines a function named 'predict_choice_proportions' that takes a single argument.")