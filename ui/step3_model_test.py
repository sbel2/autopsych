# File: ui/step3_model_test.py

import streamlit as st
import pandas as pd
import numpy as np

def render_step3():
    """Renders the UI for Step 3: Model Testing and Accuracy."""
    st.header("Step 3: Test the Model's Accuracy")
    st.info("The generated Python function is now applied to each problem from your dataset to measure its accuracy.")

    with st.expander("🐍 View Final Model Code to be Executed"):
        st.code(st.session_state.model_code, language='python')

    data = st.session_state.editable_data.copy()
    model_code = st.session_state.model_code
    model_function = None

    if not model_code.strip():
        st.error("There is no model code to test. Please go back to Step 2 and generate a model.")
    else:
        try:
            exec_globals = {}
            exec(model_code, exec_globals)
            model_function = exec_globals.get('predict_choice_proportions')
        except Exception as e:
            st.error(f"Error in model code syntax: {e}")

        if model_function:
            st.success("Python model loaded successfully! Running tests...")
            predictions_A, predictions_B, results = [], [], []
            debug_logs = []

            for index, row in data.iterrows():
                try:
                    problem_id = row.get('problem_id', f'Index {index}')
                    debug_logs.append(f"--- \n**Testing Problem ID: {problem_id}**")
                    debug_logs.append("  - **Full Input Row Data:**")
                    debug_logs.append(f"    ```\n{row.to_string()}\n    ```")

                    prediction_output = model_function(row)

                    if not isinstance(prediction_output, (tuple, list)) or len(prediction_output) != 2:
                        raise TypeError("Model function must return a tuple or list of length 2.")
                    
                    pred_A, pred_B = prediction_output
                    
                    if not isinstance(pred_A, (int, float)) or not isinstance(pred_B, (int, float)):
                         raise TypeError("Both items in the returned tuple must be numbers.")

                    debug_logs.append(f"  - **Model Prediction Proportions:** `({pred_A:.2f}, {pred_B:.2f})`")

                    predicted_choice = 'A' if pred_A > pred_B else 'B' if pred_B > pred_A else 'Tie'
                    actual_choice = 'A' if row['%A'] > row['%B'] else 'B' if row['%B'] > row['%A'] else 'Tie'
                    is_correct = (predicted_choice == actual_choice)
                    results.append("✅ Correct" if is_correct else "❌ Incorrect")
                    debug_logs.append(f"  - **Result:** Predicted Winner: **{predicted_choice}**, Actual Winner: **{actual_choice}** -> {'Correct' if is_correct else 'Incorrect'}")

                    predictions_A.append(pred_A)
                    predictions_B.append(pred_B)

                except Exception as e:
                    problem_id = row.get('problem_id', f'Index {index}')
                    st.error(f"A runtime error occurred in the model function while processing problem ID '{problem_id}'. Check the debugging logs for details.")
                    debug_logs.append(f"  - **🛑 RUNTIME ERROR:** {e}")
                    predictions_A.append(np.nan)
                    predictions_B.append(np.nan)
                    results.append("Error")

            with st.expander("🔍 View Row-by-Row Testing Details", expanded=False):
                st.markdown("\n".join(debug_logs))

            results_df = data.copy()
            results_df['model_pred_A'] = predictions_A
            results_df['model_pred_B'] = predictions_B
            results_df['Result'] = results

            if results:
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)

                correct_predictions = results.count("✅ Correct")
                total_predictions = len(results)
                accuracy_percent = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                with col1:
                    st.metric(label="Classification Accuracy", value=f"{correct_predictions} / {total_predictions} ({accuracy_percent:.1f}%)")
                    st.caption("Measures how often the model predicted the correct winner.")

                try:
                    results_df['prop_A'] = pd.to_numeric(results_df['%A'], errors='coerce') / 100.0
                    results_df['prop_B'] = pd.to_numeric(results_df['%B'], errors='coerce') / 100.0
                    
                    results_df['model_pred_A'] = pd.to_numeric(results_df['model_pred_A'], errors='coerce')
                    results_df['model_pred_B'] = pd.to_numeric(results_df['model_pred_B'], errors='coerce')
                    
                    valid_results = results_df.dropna(subset=['model_pred_A', 'model_pred_B', 'prop_A', 'prop_B'])

                    mse_A = ((valid_results['prop_A'] - valid_results['model_pred_A'])**2).mean()
                    mse_B = ((valid_results['prop_B'] - valid_results['model_pred_B'])**2).mean()
                    total_mse = (mse_A + mse_B) / 2
                    
                    with col2:
                        st.metric(label="Mean Squared Error (MSE)", value=f"{total_mse:.4f}")
                        st.caption("Measures the avg. squared distance between predicted and actual choice proportions. Lower is better.")
                except (TypeError, KeyError) as e:
                     with col2:
                        st.metric(label="Mean Squared Error (MSE)", value="N/A", help=f"Error: {e}")
                        st.caption("Could not be calculated.")

            st.subheader("Detailed Results")
            st.dataframe(results_df)

        else:
            st.warning("Could not find a valid function named `predict_choice_proportions` in the provided code.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Go Back to Edit Model"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("↩ Start Over from Scratch"):
            st.session_state.clear()
            st.rerun()