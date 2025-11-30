# File: config.py

import streamlit as st

# --- Google Gemini API Configuration ---
# Use Streamlit secrets for the API key for better security.
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please set your Google API key in the Streamlit secrets manager (e.g., .streamlit/secrets.toml).")
    st.stop()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GEMINI_MODEL = "gemini-2.5-pro"  # Using the latest Gemini 2.5 Pro model

# --- Embedded CSV Data ---
EMBEDDED_CSV_DATA = """problem_id,phenomenon_name,A_outcome_1,A_prob_1,A_outcome_2,A_prob_2,A_outcome_3,A_prob_3,A_outcome_4,A_prob_4,A_outcome_5,A_prob_5,A_outcome_6,A_prob_6,A_outcome_7,A_prob_7,A_outcome_8,A_prob_8,A_outcome_9,A_prob_9,A_outcome_10,A_prob_10,B_outcome_1,B_prob_1,B_outcome_2,B_prob_2,B_outcome_3,B_prob_3,B_outcome_4,B_prob_4,B_outcome_5,B_prob_5,B_outcome_6,B_prob_6,B_outcome_7,B_prob_7,B_outcome_8,B_prob_8,B_outcome_9,B_prob_9,B_outcome_10,B_prob_10,%A,%B
1a,Certainty effect/Allais paradox,3000.0,1.0,,,,,,,,,,,,,,,,,,,4000.0,0.8,0.0,0.2,,,,,,,,,,,,,,,,,80,20
1b,Certainty effect/Allais paradox,3000.0,0.25,0.0,0.75,,,,,,,,,,,,,,,,,4000.0,0.2,0.0,0.8,,,,,,,,,,,,,,,,,35,65
2a,Reflection effect,3000.0,1.0,,,,,,,,,,,,,,,,,,,4000.0,0.8,0.0,0.2,,,,,,,,,,,,,,,,,80,20
2b,Reflection effect,-3000.0,1.0,,,,,,,,,,,,,,,,,,,-4000.0,0.8,0.0,0.2,,,,,,,,,,,,,,,,,8,92
3,Over-weighting of rare events,5.0,1.0,,,,,,,,,,,,,,,,,,,5000.0,0.001,0.0,0.999,,,,,,,,,,,,,,,,,28,72
4,Loss aversion,0.0,1.0,,,,,,,,,,,,,,,,,,,-100.0,0.5,100.0,0.5,,,,,,,,,,,,,,,,,78,22
5,St. Petersburg paradox,9.0,1.0,,,,,,,,,,,,,,,,,,,2.0,0.5,4.0,0.25,8.0,0.125,16.0,0.0625,32.0,0.03125,64.0,0.015625,128.0,0.0078125,256.0,0.0078125,,,,,62,38
6,Ellsberg paradox,10.0,0.5,0.0,0.5,,,,,,,,,,,,,,,,,10.0,p,0.0,p,,,,,,,,,,,,,,,,,63,37
7,Low magnitude eliminates loss aversion,0.0,1.0,,,,,,,,,,,,,,,,,,,-10.0,0.5,10.0,0.5,,,,,,,,,,,,,,,,,52,48
8a,Break-even effect,-2.25,1.0,,,,,,,,,,,,,,,,,,,-4.5,0.5,0.0,0.5,,,,,,,,,,,,,,,,,13,87
8b,Break-even effect,-7.5,1.0,,,,,,,,,,,,,,,,,,,-5.25,0.5,-9.75,0.5,,,,,,,,,,,,,,,,,23,77
9a,Get-something effect,11.0,0.5,3.0,0.5,,,,,,,,,,,,,,,,,13.0,0.5,0.0,0.5,,,,,,,,,,,,,,,,,79,21
9b,Get-something effect,12.0,0.5,4.0,0.5,,,,,,,,,,,,,,,,,14.0,0.5,1.0,0.5,,,,,,,,,,,,,,,,,62,38
10,Splitting effect,96.0,0.9,14.0,0.05,12.0,0.05,,,,,,,,,,,,,,,96.0,0.85,90.0,0.05,12.0,0.1,,,,,,,,,,,,,,,27,73
11a,Under-weighting of rare events,3.0,1.0,,,,,,,,,,,,,,,,,,,32.0,0.1,0.0,0.9,,,,,,,,,,,,,,,,,68,32
11b,Under-weighting of rare events,-3.0,1.0,,,,,,,,,,,,,,,,,,,-32.0,0.1,0.0,0.9,,,,,,,,,,,,,,,,,39,61
12a,Reversed reflection,3.0,1.0,,,,,,,,,,,,,,,,,,,4.0,0.8,0.0,0.2,,,,,,,,,,,,,,,,,37,63
12b,Reversed reflection,-3.0,1.0,,,,,,,,,,,,,,,,,,,-4.0,0.8,0.0,0.2,,,,,,,,,,,,,,,,,60,40
13a,Payoff variability effect,0.0,1.0,,,,,,,,,,,,,,,,,,,1.0,1,,,,,,,,,,,,,,,,,,,4,96
13b,Payoff variability effect,0.0,1.0,,,,,,,,,,,,,,,,,,,-9.0,0.5,11.0,0.5,,,,,,,,,,,,,,,,,42,58
14a,Correlation effect,6.0,0.5,0.0,0.5,,,,,,,,,,,,,,,,,9.0,0.5,0.0,0.5,,,,,,,,,,,,,,,,,16,84
14b,Correlation effect,6.0,0.5,0.0,0.5,,,,,,,,,,,,,,,,,8.0,0.5,0.0,0.5,,,,,,,,,,,,,,,,,2,98
"""

# --- Prompt Template ---

GENERAL_PROMPT = """
You are an expert data scientist.
Your mission is to write a Python function that predicts human choices from a dataset of risky decision problems.
Here is a sample of the data you'll be working with:

{{data_summary}}
**Output format rule:**
1. Return only valid Python code.
2. Start directly with import statements.
3. Do not include any explanation, comments, or markdown formatting.

Each dataset row describes two choice options (A and B), each defined by several (outcome, probability) pairs:
- A has up to 10 pairs: A_outcome_1, A_prob_1, …, A_outcome_10, A_prob_10.
- B has up to 10 pairs: B_outcome_1, B_prob_1, …, B_outcome_10, B_prob_10. 
- Some probability cells may contain symbols like 'p'; treat them as invalid and ignore them.

Your function must be named predict_choice_proportions(problem).
The argument problem will be a dictionary or pandas Series representing one row.

**Requirements:**
1. Safely extract all numeric (outcome, probability) pairs for A and B using pd.to_numeric(..., errors='coerce'). Skip any invalid pairs (NaN outcomes or probabilities).
2. Compute a probabilistic human choice model — not a simple expected value model.
3. Convert the resulting values into a softmax-style probability of choosing A vs. B.
4. Return a tuple (prob_a, prob_b).

Here is an example model. DO NOT simply copy the example. Your goal is to create a model that has a lower error than the example when tested against the '%A', ‘%B’ data.

import pandas as pd
import numpy as np

def predict_choice_proportions(problem):
    
    # 1. Initialize accumulators
    ev_a = 0.0
    ev_b = 0.0

    # 2. Process Option A
    for i in range(1, 11):
        outcome_key = f'A_outcome_{i}'
        prob_key = f'A_prob_{i}'

        outcome_val = problem.get(outcome_key)
        prob_val = problem.get(prob_key)

        outcome_num = pd.to_numeric(outcome_val, errors='coerce')
        prob_num = pd.to_numeric(prob_val, errors='coerce')

        if pd.notna(outcome_num) and pd.notna(prob_num):
            ev_a += outcome_num * prob_num

    # 3. Process Option B
    for i in range(1, 11):
        outcome_key = f'B_outcome_{i}'
        prob_key = f'B_prob_{i}'

        outcome_val = problem.get(outcome_key)
        prob_val = problem.get(prob_key)

        outcome_num = pd.to_numeric(outcome_val, errors='coerce')
        prob_num = pd.to_numeric(prob_val, errors='coerce')

        if pd.notna(outcome_num) and pd.notna(prob_num):
            ev_b += outcome_num * prob_num

    # 4. Define beta and calculate softmax probabilities
    beta = 0.005
    
    # Handle potential overflow in np.exp by clipping the difference
    ev_diff = beta * (ev_b - ev_a)
    
    # The softmax formula for P(A)
    prob_a = 1 / (1 + np.exp(ev_diff))
    prob_b = 1 - prob_a

    return (prob_a, prob_b)

**Important:**
1. Your code must be complete and syntactically correct Python.
2. Your output must contain code only.
3. Stop after returning (prob_a, prob_b).

"""