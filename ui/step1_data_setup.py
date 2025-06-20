# File: ui/step1_data_setup.py

import streamlit as st
import pandas as pd
import io
from data_utils import generate_gambling_problems, convert_df_to_csv
from config import EMBEDDED_CSV_DATA

def render_step1():
    """Renders the UI for Step 1: Data Source Selection and Editing."""
    st.header("Step 1: Choose Your Data Source")
    st.info("Start with the pre-loaded sample data, add new synthetic data, or upload your own CSV file.")

    if st.session_state.editable_data is None:
        st.session_state.editable_data = pd.read_csv(io.StringIO(EMBEDDED_CSV_DATA))

    source_choice = st.radio(
        "Select data source:",
        ["Use Pre-loaded Sample Data", "Add New Synthetic Data", "Upload Your Own CSV File"],
        horizontal=True, key="source_choice"
    )

    if source_choice == "Use Pre-loaded Sample Data":
        if st.button("Reset to Pre-loaded Data"):
            st.session_state.editable_data = pd.read_csv(io.StringIO(EMBEDDED_CSV_DATA))
            st.rerun()

    elif source_choice == "Add New Synthetic Data":
        n_problems = st.number_input("How many problems to generate and add?", min_value=1, max_value=500, value=10, step=1)
        if st.button("Add New Synthetic Data", type="primary"):
            new_problems_df = generate_gambling_problems(n_problems)
            current_df = st.session_state.editable_data.copy()
            st.session_state.editable_data = pd.concat([current_df, new_problems_df], ignore_index=True)
            st.session_state.model_code = ""
            st.success(f"Added {n_problems} new synthetic problems to the dataset.")
            st.rerun()

    else: # Upload Your Own CSV File
        uploaded_file = st.file_uploader("Choose a CSV file to replace current data", type="csv")
        if uploaded_file is not None:
            try:
                st.session_state.editable_data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully! Current data has been replaced.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.session_state.editable_data is not None:
        st.subheader("Current Dataset")
        st.markdown(f"Review and edit the data below. The dataset currently has **{len(st.session_state.editable_data)}** problems.")

        st.session_state.editable_data = st.data_editor(st.session_state.editable_data, num_rows="dynamic", key="data_editor")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label="📥 Download Data as CSV", data=convert_df_to_csv(st.session_state.editable_data), file_name="edited_decision_data.csv", mime='text/csv')
        with col2:
            if st.button("Use this data to Generate Model →", type="primary"):
                st.session_state.step = 2
                st.rerun()