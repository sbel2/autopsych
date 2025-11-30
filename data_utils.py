# File: data_utils.py

import pandas as pd
import numpy as np

def generate_gambling_problems(n_problems):
    """
    Generates a DataFrame of n gambling problems in the new wide format.
    This version is corrected to be compatible with the main dataset.
    """
    data = []
    cols = [
        "problem_id", "phenomenon_name",
        "A_outcome_1", "A_prob_1", "A_outcome_2", "A_prob_2", "A_outcome_3", "A_prob_3",
        "A_outcome_4", "A_prob_4", "A_outcome_5", "A_prob_5", "A_outcome_6", "A_prob_6",
        "A_outcome_7", "A_prob_7", "A_outcome_8", "A_prob_8", "A_outcome_9", "A_prob_9",
        "A_outcome_10", "A_prob_10",
        "B_outcome_1", "B_prob_1", "B_outcome_2", "B_prob_2", "B_outcome_3", "B_prob_3",
        "B_outcome_4", "B_prob_4", "B_outcome_5", "B_prob_5", "B_outcome_6", "B_prob_6",
        "B_outcome_7", "B_prob_7", "B_outcome_8", "B_prob_8", "B_outcome_9", "B_prob_9",
        "B_outcome_10", "B_prob_10",
        "%A", "%B"
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

def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')