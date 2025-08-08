"""
Module for testing the predictive model's accuracy.
"""
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np

def test_model(model_code: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Test the model's predictions against the provided dataset.
    
    Args:
        model_code: String containing the Python code of the model
        data: DataFrame containing the test data
        
    Returns:
        Tuple containing:
            - DataFrame with test results
            - Dictionary with performance metrics
            - List of debug logs
    """
    debug_logs = []
    predictions_A, predictions_B, results = [], [], []
    model_function = None
    metrics = {}
    
    # Prepare results DataFrame
    results_df = data.copy()
    
    try:
        # Execute the model code to get the prediction function
        exec_globals = {}
        exec(model_code, exec_globals)
        model_function = exec_globals.get('predict_choice_proportions')
        
        if not model_function:
            raise ValueError("Could not find 'predict_choice_proportions' function in the model code.")
            
        # Test each row in the dataset
        for index, row in data.iterrows():
            try:
                problem_id = row.get('problem_id', f'Index {index}')
                debug_logs.append(f"--- \n**Testing Problem ID: {problem_id}**")
                debug_logs.append("  - **Full Input Row Data:**")
                debug_logs.append(f"    ```\n{row.to_string()}\n    ```")

                # Get model prediction
                prediction_output = model_function(row)

                # Validate prediction format
                if not isinstance(prediction_output, (tuple, list)) or len(prediction_output) != 2:
                    raise TypeError("Model function must return a tuple or list of length 2.")
                
                pred_A, pred_B = prediction_output
                
                if not isinstance(pred_A, (int, float)) or not isinstance(pred_B, (int, float)):
                    raise TypeError("Both items in the returned tuple must be numbers.")

                debug_logs.append(f"  - **Model Prediction Proportions:** `({pred_A:.2f}, {pred_B:.2f})`")

                # Determine predicted and actual choices
                predicted_choice = 'A' if pred_A > pred_B else 'B' if pred_B > pred_A else 'Tie'
                actual_choice = 'A' if row['%A'] > row['%B'] else 'B' if row['%B'] > row['%A'] else 'Tie'
                is_correct = (predicted_choice == actual_choice)
                
                results.append("✅ Correct" if is_correct else "❌ Incorrect")
                debug_logs.append(
                    f"  - **Result:** Predicted Winner: **{predicted_choice}**, "
                    f"Actual Winner: **{actual_choice}** -> {'Correct' if is_correct else 'Incorrect'}"
                )

                predictions_A.append(pred_A)
                predictions_B.append(pred_B)

            except Exception as e:
                problem_id = row.get('problem_id', f'Index {index}')
                debug_logs.append(f"  - **🛑 RUNTIME ERROR:** {e}")
                predictions_A.append(np.nan)
                predictions_B.append(np.nan)
                results.append("Error")
        
        # Add predictions to results DataFrame
        results_df['model_pred_A'] = predictions_A
        results_df['model_pred_B'] = predictions_B
        results_df['Result'] = results
        
        # Calculate metrics
        if results:
            # Classification accuracy
            correct_predictions = results.count("✅ Correct")
            total_predictions = len(results)
            metrics['accuracy_percent'] = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            metrics['correct_predictions'] = correct_predictions
            metrics['total_predictions'] = total_predictions
            
            # Calculate MSE if possible
            try:
                results_df['prop_A'] = pd.to_numeric(results_df['%A'], errors='coerce') / 100.0
                results_df['prop_B'] = pd.to_numeric(results_df['%B'], errors='coerce') / 100.0
                
                valid_results = results_df.dropna(subset=['model_pred_A', 'model_pred_B', 'prop_A', 'prop_B'])
                
                if not valid_results.empty:
                    mse_A = ((valid_results['prop_A'] - valid_results['model_pred_A'])**2).mean()
                    mse_B = ((valid_results['prop_B'] - valid_results['model_pred_B'])**2).mean()
                    metrics['mse_A'] = mse_A
                    metrics['mse_B'] = mse_B
                    metrics['total_mse'] = (mse_A + mse_B) / 2
                else:
                    metrics['error'] = "No valid results for MSE calculation"
            except (TypeError, KeyError) as e:
                metrics['error'] = f"Error calculating MSE: {str(e)}"
        
        return results_df, metrics, debug_logs
        
    except Exception as e:
        # Handle any errors in the test setup
        debug_logs.append(f"🛑 TEST SETUP ERROR: {str(e)}")
        metrics['error'] = str(e)
        return results_df, metrics, debug_logs
