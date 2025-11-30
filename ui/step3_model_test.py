# File: ui/step3_model_test.py

import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to the path to import the testing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testing import test_model

def render_step3():
    """Renders the UI for Step 3: Model Testing and Accuracy."""
    st.header("Step 3: Test the Model's Accuracy")
    st.info("The generated Python function is now applied to each problem from your dataset to measure its accuracy.")

    with st.expander("🐍 View Final Model Code to be Executed"):
        st.code(st.session_state.model_code, language='python')

    # For evaluation, always use the fixed CPC18 test dataset from the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpc18_path = os.path.join(project_root, "cpc18_test.csv")
    data = pd.read_csv(cpc18_path)
    model_code = st.session_state.model_code

    if not model_code.strip():
        st.error("There is no model code to test. Please go back to Step 2 and generate a model.")
    else:
        # Run the tests using the testing module
        results_df, metrics, debug_logs = test_model(model_code, data)
        
        # Display debug logs
        with st.expander("🔍 View Row-by-Row Testing Details", expanded=False):
            st.markdown("\n".join(debug_logs))
        
        # Display error if any occurred during test setup
        if 'error' in metrics:
            st.error(f"Error during testing: {metrics['error']}")
        # Only show results if we have valid data
        elif not results_df.empty:
            # Display model performance metrics
            st.subheader("Model Performance")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            # Classification accuracy
            with col1:
                if 'accuracy_percent' in metrics:
                    st.metric(
                        label="Classification Accuracy", 
                        value=f"{metrics['correct_predictions']} / {metrics['total_predictions']} "
                              f"({metrics['accuracy_percent']:.1f}%)"
                    )
                    st.caption("Measures how often the model predicted the correct winner.")
            
            # MSE if available
            with col2:
                if 'total_mse' in metrics:
                    st.metric(
                        label="Mean Squared Error (MSE)", 
                        value=f"{metrics['total_mse']:.4f}"
                    )
                    st.caption("Measures the avg. squared distance between predicted and actual choice proportions. Lower is better.")
                elif 'error' in metrics:
                    st.metric(
                        label="Mean Squared Error (MSE)", 
                        value="N/A", 
                        help=f"Error: {metrics['error']}"
                    )
                    st.caption("Could not be calculated.")
            
            # Display detailed results
            st.subheader("Detailed Results")
            st.dataframe(results_df)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Go Back to Edit Model"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("↩ Start Over from Scratch"):
            st.session_state.clear()
            st.rerun()