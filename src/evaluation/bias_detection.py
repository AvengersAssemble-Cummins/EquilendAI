import pandas as pd
import numpy as np
from aequitas.group import Group
from aequitas.bias import Bias
import os

def run_bias_audit(data_path, label_col='label_value', score_col='score', protected_attributes=None):
    """
    Calculates Disparate Impact and other fairness metrics using Aequitas.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    # Load the predictions data
    df = pd.read_csv(data_path)
    
    if protected_attributes is None:
        # Defaulting to common protected classes if none provided
        protected_attributes = ['age', 'location']

    # Ensure required columns exist
    required = [label_col, score_col] + protected_attributes
    for col in required:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return

    print(f"--- Starting Bias Audit on attributes: {protected_attributes} ---")

    # 1. Group() class: Calculates absolute metrics (Confusion Matrix per group)
    g = Group()
    xtab, _ = g.get_crosstabs(df, score_cols=[score_col], attr_cols=protected_attributes)

    # 2. Bias() class: Calculates disparities (ratios compared to reference groups)
    b = Bias()
    
    # We define reference groups (privileged groups). 
    # Example: 'Adult' for age, 'Urban' for location. 
    # You can adjust these based on your specific dataset labels.
    ref_groups = {
        'age': 'Adult',
        'location': 'Urban'
    }
    
    # Calculate disparities relative to reference groups
    bdf = b.get_disparity_predefined_groups(xtab, df, ref_groups_dict=ref_groups, alpha=0.05)

    # 3. Focus on Disparate Impact (Statistical Parity Disparity)
    # The 80% rule: Disparate impact should be between 0.8 and 1.25
    print("\n[DISPARATE IMPACT SUMMARY]")
    for index, row in bdf.iterrows():
        attribute = row['attribute_name']
        group = row['attribute_value']
        di_ratio = row['pprev_disparity'] # This is the Statistical Parity / Disparate Impact ratio
        
        status = "✅ PASS" if 0.8 <= di_ratio <= 1.25 else "❌ FAIL (Bias Detected)"
        print(f"Group: {attribute}={group} | Ratio: {di_ratio:.3f} | {status}")

    # Export report
    report_path = "Fairness_Report.md"
    with open(report_path, "w") as f:
        f.write("# Model Fairness & Bias Report\n")
        f.write(f"Generated Audit for: {data_path}\n\n")
        f.write("## Disparate Impact Metrics\n")
        f.write(bdf[['attribute_name', 'attribute_value', 'pprev_disparity']].to_markdown())
    
    print(f"\nAudit complete. Detailed report saved to {report_path}")

if __name__ == "__main__":
    # Assuming you have a CSV with 'score' (prediction) and 'label_value' (actual)
    # Replace 'test_results.csv' with your actual model output file
    run_bias_audit("data/model_results.csv")