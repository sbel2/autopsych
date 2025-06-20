import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import json
import re

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive Decision Model Pipeline",
    page_icon="🧠",
    layout="wide"
)

# --- Groq API Configuration ---
# Use Streamlit secrets for the API key for better security.
if "GROQ_API_KEY" not in st.secrets:
    st.error("Please set your Groq API key in the Streamlit secrets manager (e.g., .streamlit/secrets.toml).")
    st.stop()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

#6,"Ellsberg paradox",10,0.5,0,0.5,,,10,p,0,p,,,,,,,,,,,,,63,37

# --- Embedded CSV Data ---
EMBEDDED_CSV_DATA = """problem_id,phenomenon_name,A_outcome_1,A_prob_1,A_outcome_2,A_prob_2,A_outcome_3,A_prob_3,B_outcome_1,B_prob_1,B_outcome_2,B_prob_2,B_outcome_3,B_prob_3,B_outcome_4,B_prob_4,B_outcome_5,B_prob_5,B_outcome_6,B_prob_6,B_outcome_7,B_prob_7,B_outcome_8,B_prob_8,%A,%B
1a,"Certainty effect/Allais paradox",3000,1,,,,,4000,0.8,0,0.2,,,,,,,,,,,,,80,20
1b,"Certainty effect/Allais paradox",3000,0.25,0,0.75,,,4000,0.2,0,0.8,,,,,,,,,,,,,35,65
2a,"Reflection effect",3000,1,,,,,4000,0.8,0,0.2,,,,,,,,,,,,,80,20
2b,"Reflection effect",-3000,1,,,,,-4000,0.8,0,0.2,,,,,,,,,,,,,8,92
3,"Over-weighting of rare events",5,1,,,,,5000,0.001,0,0.999,,,,,,,,,,,,,28,72
4,"Loss aversion",0,1,,,,,-100,0.5,100,0.5,,,,,,,,,,,,,78,22
5,"St. Petersburg paradox",9,1,,,,,2,0.5,4,0.25,8,0.125,16,0.0625,32,0.03125,64,0.015625,128,0.0078125,256,0.0078125,62,38
7,"Low magnitude eliminates loss aversion",0,1,,,,,-10,0.5,10,0.5,,,,,,,,,,,,,52,48
8a,"Break-even effect",-2.25,1,,,,,-4.5,0.5,0,0.5,,,,,,,,,,,,,13,87
8b,"Break-even effect",-7.5,1,,,,,-5.25,0.5,-9.75,0.5,,,,,,,,,,,,,23,77
9a,"Get-something effect",11,0.5,3,0.5,,,13,0.5,0,0.5,,,,,,,,,,,,,79,21
9b,"Get-something effect",12,0.5,4,0.5,,,14,0.5,1,0.5,,,,,,,,,,,,,62,38
10,"Splitting effect",96,0.9,14,0.05,12,0.05,96,0.85,90,0.05,12,0.1,,,,,,,,,,,27,73
11a,"Under-weighting of rare events",3,1,,,,,32,0.1,0,0.9,,,,,,,,,,,,,68,32
11b,"Under-weighting of rare events",-3,1,,,,,-32,0.1,0,0.9,,,,,,,,,,,,,39,61
12a,"Reversed reflection",3,1,,,,,4,0.8,0,0.2,,,,,,,,,,,,,37,63
12b,"Reversed reflection",-3,1,,,,,-4,0.8,0,0.2,,,,,,,,,,,,,60,40
13a,"Payoff variability effect",0,1,,,,,1,1,,,,,,,,,,,,,,,4,96
13b,"Payoff variability effect",0,1,,,,,-9,0.5,11,0.5,,,,,,,,,,,,,42,58
14a,"Correlation effect",6,0.5,0,0.5,,,9,0.5,0,0.5,,,,,,,,,,,,,16,84
14b,"Correlation effect",6,0.5,0,0.5,,,8,0.5,0,0.5,,,,,,,,,,,,,2,98
"""

# --- Prompt Template ---

GENERAL_PROMPT = """
You are an expert data scientist. Your mission is to write a Python function that can predict the choices recorded in a dataset.

**Your Task:**
Create a Python function named `predict_choice_proportions`. This function should accept a single argument, `problem`, which will be a dictionary or pandas Series representing one row of the dataset. Add any import statements needed to make the model work. Do not include anything other than plain Python code. We are inputting your response straight into a model tester, so any verbal response will only trigger errors. Do not say "here is the python function". Please make sure there are no syntax errors in the code.

**Function Inputs (The Ingredients):**
The `problem` input your function receives is a dictionary or pandas Series containing the full payoff structure for two choices.

**A Step-by-Step Guide to Processing the Input:**
To write a stable function, you must follow this exact procedure to parse the outcomes and probabilities for each option:

1.  **Initialize empty lists:** Create four lists: `A_outcomes`, `A_probs`, `B_outcomes`, `B_probs`.
2.  **Loop Explicitly by Number:**
    * To get Option A's data, loop through numbers `i` from 1 to 3.
    * To get Option B's data, loop through numbers `i` from 1 to 8.
3.  **Construct Keys and Access Data Safely:** Inside your loop, create the column name strings (e.g., `f'A_outcome_i'` and `f'A_prob_i'`). Use the `.get()` method to access the data from the `problem` object to avoid errors if a key is missing.
4.  **Validate and Clean Data:** The values you retrieve might be missing (`NaN`) or non-numeric (e.g., the string 'p').
    * You **must** convert these values to a numeric format. The most robust way is with `pd.to_numeric(value, errors='coerce')`, which turns invalid data into `NaN`.
    * After converting, check if **both** the outcome and its corresponding probability are valid numbers using `pd.notna()`.
5.  **Append to Lists:** Only if both the outcome and probability are valid numbers should you append them to their respective lists.

This procedure will give you four clean lists of numbers to use for your calculations.

**Function Output (The Final Answer):**
After its calculation, the function must return a single tuple containing a matched pair of two numbers representing the predicted proportions for A and B, respectively.

**Your Objective: Build a Probabilistic Model**
Your goal is to create a model that reflects the noisy, probabilistic nature of human choice.

Return **ONLY** the complete and runnable Python code for the `predict_choice_proportions` function that implements this probabilistic logic.

--- DATA FOR ANALYSIS ---
{data_summary}
--- END DATA ---
"""

# --- Helper Functions ---

def call_groq_api(prompt):
    """Makes a live API call to the Groq service to generate the model, with debugging and safe code cleanup."""
    st.info("Preparing to call Groq API...")
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    with st.expander("🔍 View API Request Details"):
        st.write("Sending the following payload to Groq:")
        st.json(payload)

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
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



def generate_gambling_problems(n_problems):
    """
    Generates a DataFrame of n gambling problems in the new wide format.
    This version is corrected to be compatible with the main dataset.
    """
    data = []
    cols = [
        "problem_id", "phenomenon_name", "A_outcome_1", "A_prob_1", "A_outcome_2",
        "A_prob_2", "A_outcome_3", "A_prob_3", "B_outcome_1", "B_prob_1",
        "B_outcome_2", "B_prob_2", "B_outcome_3", "B_prob_3", "B_outcome_4",
        "B_prob_4", "B_outcome_5", "B_prob_5", "B_outcome_6", "B_prob_6",
        "B_outcome_7", "B_prob_7", "B_outcome_8", "B_prob_8", "%A", "%B"
    ]

    for i in range(n_problems):
        problem_row = {col: np.nan for col in cols}

        v_A1 = np.random.randint(10, 100)
        p_A1 = np.random.uniform(0.2, 0.9)
        v_A2 = 0
        p_A2 = 1 - p_A1
        ev_A = v_A1 * p_A1

        v_B1 = np.random.randint(10, 100)
        p_B1 = np.random.uniform(0.2, 0.9)
        v_B2 = 0
        p_B2 = 1 - p_B1
        ev_B = v_B1 * p_B1

        if abs(ev_A - ev_B) < 0.01:
            prob_A, prob_B = 50, 50
        elif ev_A > ev_B:
            prob_A, prob_B = 100, 0
        else:
            prob_A, prob_B = 0, 100

        problem_row.update({
            "problem_id": f"gen_{i+1}",
            "phenomenon_name": "Synthetic EV Choice",
            "A_outcome_1": v_A1, "A_prob_1": round(p_A1, 2),
            "A_outcome_2": v_A2, "A_prob_2": round(p_A2, 2),
            "B_outcome_1": v_B1, "B_prob_1": round(p_B1, 2),
            "B_outcome_2": v_B2, "B_prob_2": round(p_B2, 2),
            "%A": prob_A,
            "%B": prob_B
        })
        data.append(problem_row)

    return pd.DataFrame(data)


@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- App State Management ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'editable_data' not in st.session_state: st.session_state.editable_data = None
if 'llm_prompt' not in st.session_state: st.session_state.llm_prompt = ""
if 'model_code' not in st.session_state: st.session_state.model_code = ""

# --- Main App UI ---
st.title("🧠 Human Decision Model Iteration Pipeline")
st.markdown("---")

# --- STEP 1: Select or Generate Data ---
if st.session_state.step == 1:
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

# --- STEP 2: Generate Python Model from Data ---
if st.session_state.step == 2:
    st.header("Step 2: Generate a Predictive Model with an LLM")
    st.info("Review the prompt below, edit it if needed, and then generate a model with the LLM.")

    prompt_template = GENERAL_PROMPT
    data_summary_text = st.session_state.get('editable_data', pd.DataFrame()).head(10).to_string()
    prompt_text = prompt_template.format(data_summary=data_summary_text)

    st.subheader("Editable LLM Prompt")
    st.session_state.llm_prompt = st.text_area("LLM Prompt:", value=prompt_text, height=400)

    if st.button("Generate Python Model via API", type="primary"):
        with st.spinner("Calling Groq API... Please wait for the live result."):
            generated_code = call_groq_api(st.session_state.llm_prompt)
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

# --- STEP 3: Test Model and View Accuracy ---
if st.session_state.step == 3:
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

                    # Execute the model function
                    prediction_output = model_function(row)

                    # --- ADDED VALIDATION for stability ---
                    if not isinstance(prediction_output, (tuple, list)) or len(prediction_output) != 2:
                        raise TypeError("Model function must return a tuple or list of length 2.")
                    
                    pred_A, pred_B = prediction_output
                    
                    if not isinstance(pred_A, (int, float)) or not isinstance(pred_B, (int, float)):
                         raise TypeError("Both items in the returned tuple must be numbers.")
                    # --- END VALIDATION ---

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