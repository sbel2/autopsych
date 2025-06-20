# File: app.py

import streamlit as st
import pandas as pd
import io

# Import the UI rendering functions from their respective modules
from ui.step1_data_setup import render_step1
from ui.step2_model_gen import render_step2
from ui.step3_model_test import render_step3
from config import EMBEDDED_CSV_DATA

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive Decision Model Pipeline",
    page_icon="🧠",
    layout="wide"
)

# --- App State Management ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'editable_data' not in st.session_state:
    # Initialize with the pre-loaded data
    st.session_state.editable_data = pd.read_csv(io.StringIO(EMBEDDED_CSV_DATA))
if 'llm_prompt' not in st.session_state:
    st.session_state.llm_prompt = ""
if 'model_code' not in st.session_state:
    st.session_state.model_code = ""

# --- Main App UI ---
st.title("🧠 Human Decision Model Iteration Pipeline")
st.markdown("---")

# --- Page routing based on step ---
if st.session_state.step == 1:
    render_step1()
elif st.session_state.step == 2:
    render_step2()
elif st.session_state.step == 3:
    render_step3()