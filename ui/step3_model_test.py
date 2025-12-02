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
    st.info("The generated Python function is now applied to each problem from both the training and test datasets to measure its accuracy.")

    with st.expander("🐍 View Final Model Code to be Executed"):
        st.code(st.session_state.model_code, language='python')

    model_code = st.session_state.model_code

    if not model_code.strip():
        st.error("There is no model code to test. Please go back to Step 2 and generate a model.")
    else:
        # Get training data from session state
        train_data = st.session_state.editable_data
        
        # Get test data from the fixed CPC18 test dataset
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cpc18_path = os.path.join(project_root, "cpc18_test.csv")
        test_data = pd.read_csv(cpc18_path)
        
        # Evaluate on Training Data
        st.subheader("📊 Training Data Evaluation")
        st.markdown("Evaluating the model on the data it was trained on:")
        
        train_results_df, train_metrics, train_debug_logs = test_model(model_code, train_data)
        
        # Display training data error if any occurred during test setup
        if 'error' in train_metrics:
            st.error(f"Error during training data testing: {train_metrics['error']}")
        # Only show results if we have valid data
        elif not train_results_df.empty:
            # Display model performance metrics for training data
            col1, col2 = st.columns(2)
            
            # Classification accuracy
            with col1:
                if 'accuracy_percent' in train_metrics:
                    st.metric(
                        label="Classification Accuracy (Training)", 
                        value=f"{train_metrics['correct_predictions']} / {train_metrics['total_predictions']} "
                              f"({train_metrics['accuracy_percent']:.1f}%)"
                    )
                    st.caption("Measures how often the model predicted the correct winner.")
            
            # MSE if available
            with col2:
                if 'total_mse' in train_metrics:
                    st.metric(
                        label="Mean Squared Error (Training)", 
                        value=f"{train_metrics['total_mse']:.4f}"
                    )
                    st.caption("Measures the avg. squared distance between predicted and actual choice proportions. Lower is better.")
                elif 'error' in train_metrics:
                    st.metric(
                        label="Mean Squared Error (Training)", 
                        value="N/A", 
                        help=f"Error: {train_metrics['error']}"
                    )
                    st.caption("Could not be calculated.")
            
            # Display detailed results for training data
            with st.expander("📋 View Detailed Training Results", expanded=False):
                st.dataframe(train_results_df)
            
            # Display debug logs for training data
            with st.expander("🔍 View Row-by-Row Training Testing Details", expanded=False):
                st.markdown("\n".join(train_debug_logs))
        
        st.divider()
        
        # Evaluate on Test Data
        st.subheader("🧪 Test Data Evaluation")
        st.markdown("Evaluating the model on held-out test data:")
        
        test_results_df, test_metrics, test_debug_logs = test_model(model_code, test_data)
        
        # Display test data error if any occurred during test setup
        if 'error' in test_metrics:
            st.error(f"Error during test data testing: {test_metrics['error']}")
        # Only show results if we have valid data
        elif not test_results_df.empty:
            # Display model performance metrics for test data
            col1, col2 = st.columns(2)
            
            # Classification accuracy
            with col1:
                if 'accuracy_percent' in test_metrics:
                    st.metric(
                        label="Classification Accuracy (Test)", 
                        value=f"{test_metrics['correct_predictions']} / {test_metrics['total_predictions']} "
                              f"({test_metrics['accuracy_percent']:.1f}%)"
                    )
                    st.caption("Measures how often the model predicted the correct winner.")
            
            # MSE if available
            with col2:
                if 'total_mse' in test_metrics:
                    st.metric(
                        label="Mean Squared Error (Test)", 
                        value=f"{test_metrics['total_mse']:.4f}"
                    )
                    st.caption("Measures the avg. squared distance between predicted and actual choice proportions. Lower is better.")
                elif 'error' in test_metrics:
                    st.metric(
                        label="Mean Squared Error (Test)", 
                        value="N/A", 
                        help=f"Error: {test_metrics['error']}"
                    )
                    st.caption("Could not be calculated.")
            
            # Display detailed results for test data
            with st.expander("📋 View Detailed Test Results", expanded=False):
                st.dataframe(test_results_df)
            
            # Display debug logs for test data
            with st.expander("🔍 View Row-by-Row Test Testing Details", expanded=False):
                st.markdown("\n".join(test_debug_logs))

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