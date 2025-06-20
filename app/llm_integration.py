# File: llm_integration.py

import streamlit as st
import requests
import json
import re

def call_groq_api(prompt, api_key, api_url):
    """Makes a live API call to the Groq service to generate the model."""
    st.info("Preparing to call Groq API...")
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # Set to 0.0 for maximum stability
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    with st.expander("🔍 View API Request Details"):
        st.write("Sending the following payload to Groq:")
        st.json(payload)

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        with st.expander("🔍 View Full API Response"):
            st.write("Received the following response from Groq:")
            st.json(result)

        model_code = result['choices'][0]['message']['content']
        model_code = re.sub(r"```(?:python)?\s*", "", model_code, flags=re.IGNORECASE)
        model_code = re.sub(r"```\s*", "", model_code)
        model_code = model_code.strip()

        st.success("API call successful. Extracted model code.")
        return model_code
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Failed to parse API response: {e}. Full response: {response.text}")
        return None