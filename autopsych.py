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
# It's better practice to use st.secrets for API keys, but using the provided one directly for this case.
if "GROQ_API_KEY" not in st.secrets:
    st.error("Please set your Groq API key in the Streamlit secrets manager.")
    st.stop()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Embedded CSV Data ---
EMBEDDED_CSV_DATA = """problem_id,p_win_A,value_A,p_win_B,value_B,sim_choice_A,sim_choice_B
1,0.22,96,0.37,13,1.0,0.0
2,0.61,45,0.73,28,1.0,0.0
3,0.69,71,0.42,73,1.0,0.0
4,0.11,81,0.22,34,1.0,0.0
5,0.85,31,0.11,96,1.0,0.0
6,0.59,49,0.29,62,1.0,0.0
7,0.35,35,0.28,21,1.0,0.0
8,0.73,81,0.34,69,1.0,0.0
9,0.7,69,0.11,62,1.0,0.0
10,0.83,69,0.81,35,1.0,0.0
"""

# --- Prompt Templates ---

CHALLENGE_PROMPT = """
You are an expert data scientist. Your mission is to write a Python function that can predict the choices recorded in a dataset.

**Your Task:**
Create a Python function named `predict_choice_proportions`. Add any import statements as well to make the model work. Do not include any explanations or markdown formatting. Just the plain Python code.

**Function Inputs (The Ingredients):**
To do its job, the function must accept four specific pieces of information for each decision problem:
* `p_win_A`: The probability of winning for Option A.
* `value_A`: The amount you get if you win with Option A.
* `p_win_B`: The probability of winning for Option B.
* `value_B`: The amount you get if you win with Option B.

**Function Output (The Final Answer):**
After its calculation, the function must produce one final answer. This answer must be a single package containing a matched pair of two numbers. For example, an output of `(1.0, 0.0)` means the model predicts 100% of choices will be for Option A.

**Your Objective: Maximize Accuracy**
Your primary goal is to make your function's predictions as close as possible to the actual choices in the data provided below. Analyze the data to find the hidden rule that connects the four inputs to the final choice. There is an answer that would reach 100% accuracy. Do not stop until you find it.

Return **ONLY** the complete and runnable Python code for the `predict_choice_proportions` function. Do not include any text or explanations before or after the code.

**Here is the data for you to analyze:**
```
{data_summary}
```
"""

GUIDED_PROMPT = """
You are a computer scientist and logician. Your mission is to write a Python function that perfectly replicates the decision rule shown in a dataset.

**Your Task:**
Create a Python function named `predict_choice_proportions`. Add any import statements as well to make the model work. Do not include any explanations or markdown formatting. Just the plain Python code.

**The Decision Agent:**
The agent you are modeling is a simple, deterministic machine. It is not a human and has no uncertainty.
* It will always make the same choice with 100% certainty.
* Its output for the chosen option is always exactly `1.0`, and for the other option, it is always exactly `0.0`.
* If both options are of equal value, it defaults to a state of indecision, represented as `(0.5, 0.5)`.

**Your Objective:**
Analyze the data below to discover the exact, hidden mathematical rule the machine uses to compare options.

To find the rule, consider how a rational machine would combine the probability and value for each option into a single "attractiveness score" to compare them. The most logical way to create this score is to multiply the probability by the value.

Implement the discovered rule in your function to achieve 100% accuracy.

Return **ONLY** the complete and runnable Python code. Do not include any explanations.

**Here is the data for you to analyze:**
```
{data_summary}
```
"""


# --- Helper Functions ---

def call_groq_api(prompt):
    """Makes a live API call to the Groq service to generate the model, with debugging and safe code cleanup."""
    st.info("Preparing to call Groq API...")
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
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

        # Get raw content from response
        model_code = result['choices'][0]['message']['content']

        # --- NEW: Remove markdown code block fences ---
        model_code = re.sub(r"```(?:python)?", "", model_code, flags=re.IGNORECASE)
        model_code = re.sub(r"```", "", model_code)
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
    """Generates a DataFrame of n gambling problems."""
    data = []
    problems_generated = 0
    while problems_generated < n_problems:
        p_win_A, value_A = np.random.uniform(0.1, 0.9), np.random.randint(10, 100)
        p_win_B, value_B = np.random.uniform(0.1, 0.9), np.random.randint(5, 100)
        ev_A, ev_B = p_win_A * value_A, p_win_B * value_B
        if ev_A == ev_B: continue
        choice_A, choice_B = (1.0, 0.0) if ev_A > ev_B else (0.0, 1.0)
        data.append({
            "problem_id": problems_generated + 1,
            "p_win_A": round(p_win_A, 2), "value_A": value_A,
            "p_win_B": round(p_win_B, 2), "value_B": value_B,
            "sim_choice_A": choice_A, "sim_choice_B": choice_B
        })
        problems_generated += 1
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
    st.info("Start with the pre-loaded sample data, generate new data, or upload your own CSV file.")

    if st.session_state.editable_data is None:
        st.session_state.editable_data = pd.read_csv(io.StringIO(EMBEDDED_CSV_DATA))

    source_choice = st.radio(
        "Select data source:",
        ["Use Pre-loaded Sample Data", "Generate New Synthetic Data", "Upload Your Own CSV File"],
        horizontal=True, key="source_choice"
    )

    if source_choice == "Use Pre-loaded Sample Data":
        st.session_state.editable_data = pd.read_csv(io.StringIO(EMBEDDED_CSV_DATA))
    elif source_choice == "Generate New Synthetic Data":
        n_problems = st.number_input("How many gambling problems to generate?", min_value=5, max_value=1000, value=10, step=5)
        if st.button("Generate Simulated Data", type="primary"):
            st.session_state.editable_data = generate_gambling_problems(n_problems)
            st.session_state.model_code = ""
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                st.session_state.editable_data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e: st.error(f"Error reading file: {e}")

    if st.session_state.editable_data is not None:
        st.subheader("Preview and Edit Data")
        st.markdown("Review the data below. You can edit values directly in the table.")
        
        st.session_state.editable_data = st.data_editor(st.session_state.editable_data, num_rows="dynamic", key="data_editor")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label="📥 Download Data as CSV", data=convert_df_to_csv(st.session_state.editable_data), file_name="edited_gambling_data.csv", mime='text/csv')
        with col2:
            if st.button("Use this data to Generate Model →", type="primary"):
                st.session_state.step = 2
                st.rerun()

# --- STEP 2: Generate Python Model from Data ---
if st.session_state.step == 2:
    st.header("Step 2: Generate a Predictive Model with an LLM")
    st.info("Select a prompt template, edit it if needed, and then generate a model with the LLM.")

    prompt_options = {
        "Challenge Prompt (Vague)": CHALLENGE_PROMPT,
        "Guided Prompt (Specific)": GUIDED_PROMPT
    }
    
    selected_prompt_name = st.selectbox(
        "Choose a prompt template:",
        options=list(prompt_options.keys())
    )
    
    prompt_template = prompt_options[selected_prompt_name]

    st.subheader("Editable LLM Prompt")
    st.session_state.llm_prompt = st.text_area("LLM Prompt:", value=prompt_template.format(data_summary=st.session_state.editable_data.to_string()), height=350)
    
    if st.button("Generate Python Model via API", type="primary"):
        with st.spinner("Calling Groq API... Please wait for the live result."):
            generated_code = call_groq_api(st.session_state.llm_prompt)
            if generated_code:
                st.session_state.model_code = generated_code
            else:
                st.error("Model generation failed. Please check the error messages in the expanders above and try again.")
    
    st.subheader("Editable Python Model")
    st.info("The LLM might generate code with minor syntax errors. Please fix them before using the model.")
    st.session_state.model_code = st.text_area("Python Model Code:", value=st.session_state.model_code, height=300, key="model_editor",
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

    data = st.session_state.editable_data
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
            predictions_A, predictions_B = [], []
            debug_logs = []
            
            for index, row in data.iterrows():
                try:
                    problem_id = row.get('problem_id', f'Index {index}')
                    debug_logs.append(f"--- \n**Testing Problem ID: {problem_id}**")
                    debug_logs.append(f"  - Input Data: p_A={row['p_win_A']}, v_A={row['value_A']}, p_B={row['p_win_B']}, v_B={row['value_B']}")
                    
                    pred_A, pred_B = model_function(row['p_win_A'], row['value_A'], row['p_win_B'], row['value_B'])
                    
                    debug_logs.append(f"  - Model Prediction: `({pred_A}, {pred_B})`")
                    predictions_A.append(pred_A)
                    predictions_B.append(pred_B)
                except Exception as e:
                    problem_id = row.get('problem_id', f'Index {index}')
                    st.error(f"Error running model on problem_id {problem_id}: {e}")
                    debug_logs.append(f"  - **ERROR**: {e}")
                    predictions_A.append(None)
                    predictions_B.append(None)

            with st.expander("🔍 View Row-by-Row Testing Details"):
                st.markdown("\n".join(debug_logs))

            results_df = data.copy()
            results_df['model_pred_A'] = predictions_A
            results_df['model_pred_B'] = predictions_B
            results_df = results_df.dropna(subset=['model_pred_A', 'model_pred_B'])

            if not results_df.empty:
                st.subheader("Model Performance")
                with st.container(border=True):
                    st.markdown("##### How is Accuracy Calculated?")
                    st.markdown("""
                    Prediction accuracy is measured using **Mean Squared Error (MSE)**. Here’s how it works:
                    1.  For each problem, we take the difference between the model's prediction and the actual observed choice proportion (e.g., `sim_choice_A` - `model_pred_A`).
                    2.  This difference is squared to make it positive and to penalize larger errors more heavily.
                    3.  We calculate the average of these squared differences across all problems.
                    
                    **A lower MSE is better**, with an MSE of **0.0** representing a perfect prediction.
                    """)
                    mse_A = ((results_df['sim_choice_A'] - results_df['model_pred_A'])**2).mean()
                    mse_B = ((results_df['sim_choice_B'] - results_df['model_pred_B'])**2).mean()
                    total_mse = (mse_A + mse_B) / 2
                    st.metric(label="Overall Model Accuracy (Mean Squared Error)", value=f"{total_mse:.4f}")

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
