# File: llm_integration.py

import streamlit as st
import google.generativeai as genai
import re

def call_gemini_api(prompt, api_key, model_name):
    """Makes a live API call to the Google Gemini service to generate the model."""
    st.info("Preparing to call Google Gemini API...")
    
    # Configure the API key
    genai.configure(api_key=api_key)
    
    # Create the model
    generation_config = {
        "temperature": 0.0,  # Set to 0.0 for maximum stability
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8192,
    }
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    with st.expander("🔍 View API Request Details"):
        st.write("Sending the following prompt to Google Gemini:")
        st.text(prompt)

    try:
        response = model.generate_content(prompt)
        
        with st.expander("🔍 View Full API Response"):
            st.write("Received the following response from Google Gemini:")
            st.json(response.__dict__)
        
        if not response.text:
            st.error("No text was generated in the response.")
            return None
            
        # Extract code from markdown code blocks if present
        model_code = response.text
        if "```" in model_code:
            model_code = re.sub(r"```(?:python)?\s*", "", model_code, flags=re.IGNORECASE)
            model_code = re.sub(r"```\s*", "", model_code)
        
        model_code = model_code.strip()

        st.success("API call successful. Extracted model code.")
        return model_code
        
    except Exception as e:
        st.error(f"API Request Failed: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            st.error(f"Error details: {e.response.text}")
        return None