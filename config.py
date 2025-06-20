# File: config.py

import streamlit as st

# --- Groq API Configuration ---
# Use Streamlit secrets for the API key for better security.
if "GROQ_API_KEY" not in st.secrets:
    st.error("Please set your Groq API key in the Streamlit secrets manager (e.g., .streamlit/secrets.toml).")
    st.stop()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Embedded CSV Data ---
EMBEDDED_CSV_DATA = """problem_id,phenomenon_name,A_outcome_1,A_prob_1,A_outcome_2,A_prob_2,A_outcome_3,A_prob_3,B_outcome_1,B_prob_1,B_outcome_2,B_prob_2,B_outcome_3,B_prob_3,B_outcome_4,B_prob_4,B_outcome_5,B_prob_5,B_outcome_6,B_prob_6,B_outcome_7,B_prob_7,B_outcome_8,B_prob_8,%A,%B
1a,"Certainty effect/Allais paradox",3000,1,,,,,4000,0.8,0,0.2,,,,,,,,,,,,,80,20
1b,"Certainty effect/Allais paradox",3000,0.25,0,0.75,,,4000,0.2,0,0.8,,,,,,,,,,,,,35,65
2a,"Reflection effect",3000,1,,,,,4000,0.8,0,0.2,,,,,,,,,,,,,80,20
2b,"Reflection effect",-3000,1,,,,,-4000,0.8,0,0.2,,,,,,,,,,,,,8,92
3,"Over-weighting of rare events",5,1,,,,,5000,0.001,0,0.999,,,,,,,,,,,,,28,72
4,"Loss aversion",0,1,,,,,-100,0.5,100,0.5,,,,,,,,,,,,,78,22
5,"St. Petersburg paradox",9,1,,,,,2,0.5,4,0.25,8,0.125,16,0.0625,32,0.03125,64,0.015625,128,0.0078125,256,0.0078125,62,38
6,"Ellsberg paradox",10,0.5,0,0.5,,,10,p,0,p,,,,,,,,,,,,,63,37
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

1.  **Initialize accumulators:** Create two variables, `ev_a` and `ev_b`, and set both to `0`.
2.  **Process Option A:** Loop through numbers `i` from 1 to 3. In each iteration:
    * Construct the keys: `outcome_key = f'A_outcome_{{i}}'` and `prob_key = f'A_prob_{{i}}'`.
    * Safely get the values from the input `problem` using `problem.get(key)`.
    * Convert both the outcome and probability values to numeric types using `pd.to_numeric(value, errors='coerce')`. This will handle non-numeric strings (like 'p') by turning them into `NaN`.
    * Check if **both** the cleaned outcome and probability are valid numbers using `pd.notna()`.
    * If and only if both are valid, multiply them together and add the result to `ev_a`.
3.  **Process Option B:** Loop through numbers `i` from 1 to 8. Repeat the exact same safe access, numeric conversion, and validation steps as you did for Option A, adding the results to `ev_b`.

**Function Output (The Final Answer):**
After its calculation, the function must return a single tuple containing a matched pair of two numbers representing the predicted proportions for A and B, respectively.

**Your Objective: Build a Probabilistic Model**
Your goal is to create a model that reflects the noisy, probabilistic nature of human choice, rather than a simple winner-take-all machine.

1.  **Calculate Expected Values (EV):** First, calculate `ev_a` and `ev_b` using the robust procedure described in the "Processing the Input" section.
2.  **Use a Softmax Function:** Convert the calculated `ev_a` and `ev_b` into choice probabilities using a softmax (Logit) function. The formula for the probability of choosing Option A is:
    `P(A) = 1 / (1 + np.exp(beta * (ev_b - ev_a)))`
3.  **Define the `beta` Parameter:** Inside your function, you must define `beta`, which represents choice sensitivity. Based on the scale of the monetary values in the data, a very small `beta` is required to produce non-extreme probabilities. **Use a `beta` value of `0.005`**.
4.  **Return Probabilities:** Your function should calculate `prob_A` using the formula, calculate `prob_B` as `1 - prob_A`, and return them in a tuple: `(prob_A, prob_B)`.

Return **ONLY** the complete and runnable Python code for the `predict_choice_proportions` function.

--- DATA FOR ANALYSIS ---
{data_summary}
--- END DATA ---
"""