"""
Sensitivity Analysis (Ablation Study) — Weight Optimization for CNRS.
Iterates through different (alpha, beta, gamma) weights to find optimal resilience.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import run_single_experiment
from config import WEIGHT_PROFILES, RANDOM_SEEDS, SCENARIO_PROFILES
import risk_scoring

def run_sensitivity_study():
    results = []
    scenario = "combined_stress"  # Best for testing robustness
    seed = RANDOM_SEEDS[0]
    
    print("="*60)
    print(f"Running Sensitivity Study on Scenario: {scenario}")
    print("="*60)
    
    for i, (a, b, g) in enumerate(WEIGHT_PROFILES):
        # Override weights globally in the module
        risk_scoring.CNRS_ALPHA = a
        risk_scoring.CNRS_BETA = b
        risk_scoring.CNRS_GAMMA = g
        
        label = f"A={a}, B={b}, G={g}"
        print(f"Profile {i+1}: {label}")
        
        history_df, _ = run_single_experiment("adaptive", scenario, seed, verbose=False)
        
        metrics = {
            'Profile': label,
            'Fork Rate': history_df['fork_occurred'].mean(),
            'Consensus Failure': 1.0 - history_df['consensus_success'].mean(),
            'Avg CNRS': history_df['cnrs'].mean()
        }
        results.append(metrics)

    df = pd.DataFrame(results)
    print("\nSensitivity Results:")
    print(df.to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_context("paper")
    
    ax = df.plot(x='Profile', y=['Fork Rate', 'Consensus Failure'], kind='bar', 
                 figsize=(12, 6), color=['#4CAF50', '#F44336'])
    plt.title(f"Ablation Study: Impact of CNRS Weights on Resilience\n({scenario} scenario)")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = os.path.join(os.path.dirname(__file__), "results", "sensitivity_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSensitivity plot saved to: {output_path}")

if __name__ == "__main__":
    run_sensitivity_study()
