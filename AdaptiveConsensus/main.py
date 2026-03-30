import pandas as pd
import numpy as np
import os
from telemetry_aggregation import aggregate_telemetry
from predictive_models import PredictiveModels
from consensus_simulator import ConsensusSimulator
from evaluation import Evaluator

def main():
    dataset_path = '/home/jobin/bct proj/BlockSim/telemetry_dataset.xlsx'
    
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    df = pd.read_excel(dataset_path)
    
    # ==========================================
    # ---> FAULT INJECTION SCENARIO START <---
    # ==========================================
    print("Injecting simulated faults for training and evaluation...")
    
    # Ensure numerical columns are represented as floats so multiplication doesn't fail on int64 columns
    float_cols = ['uptime', 'missed_vote_rate', 'vote_delay_sec', 
                  'msg_latency_ms', 'packet_loss_rate', 'partition_indicator', 'fork_occurrences', 'health_label']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
    all_validators = df['validator_id'].unique()
    num_bad = int(len(all_validators) * 0.33)
    np.random.seed(42)
    bad_validators = np.random.choice(all_validators, num_bad, replace=False)
    
    # 1. Inject Minor Faults in FIRST half (Training Data) so AI learns what failures look like
    train_fault_start = df['epoch'].min() + 10
    
    mask_train_bad = (df['epoch'] >= train_fault_start) & (df['epoch'] < train_fault_start + 10) & (df['validator_id'].isin(bad_validators))
    if 'health_label' in df.columns:
        df.loc[mask_train_bad, 'health_label'] = 0
    df.loc[mask_train_bad, 'uptime'] *= 0.1
    df.loc[mask_train_bad, 'missed_vote_rate'] += 0.8
    df.loc[mask_train_bad, 'vote_delay_sec'] *= 5.0
    
    mask_train_net = (df['epoch'] >= train_fault_start) & (df['epoch'] < train_fault_start + 10)
    df.loc[mask_train_net, 'msg_latency_ms'] *= 3.0
    df.loc[mask_train_net, 'fork_occurrences'] = 1

    # 2. Inject Catastrophic Faults in SECOND half (Test Data) to trigger Tier 3 Self-Healing
    test_fault_start = df['epoch'].median() + 20
    
    # Infect 90% of validators to push the network anomaly mean to ~0.9
    num_catastrophic = int(len(all_validators) * 0.90)
    catastrophic_validators = np.random.choice(all_validators, num_catastrophic, replace=False)
    
    mask_test_bad = (df['epoch'] >= test_fault_start) & (df['epoch'] < test_fault_start + 15) & (df['validator_id'].isin(catastrophic_validators))
    if 'health_label' in df.columns:
        df.loc[mask_test_bad, 'health_label'] = 0
    df.loc[mask_test_bad, 'uptime'] *= 0.01
    df.loc[mask_test_bad, 'missed_vote_rate'] += 0.95
    df.loc[mask_test_bad, 'vote_delay_sec'] *= 15.0
    
    mask_test_net = (df['epoch'] >= test_fault_start) & (df['epoch'] < test_fault_start + 15)
    df.loc[mask_test_net, 'msg_latency_ms'] *= 10.0
    df.loc[mask_test_net, 'fork_occurrences'] = 1
    df.loc[mask_test_net, 'partition_indicator'] = 1
    # ==========================================
    # ---> FAULT INJECTION SCENARIO END <---
    # ==========================================
    
    print("Layer 1: Aggregating Telemetry...")
    df_agg = aggregate_telemetry(df)
    
    print("Layer 2: Training Predictive Models...")
    models = PredictiveModels()
    # For a real system this would be offline training then online predicting.
    # Here we train on the first half and simulate on the second half to see adaptation.
    split_epoch = df_agg['epoch'].median()
    train_df = df_agg[df_agg['epoch'] <= split_epoch].copy()
    test_df = df_agg[df_agg['epoch'] > split_epoch].copy()

    models.train(train_df)
    
    print("Layer 3-5: Running Consensus Simulation Loop...")
    simulator = ConsensusSimulator(alpha=0.4, beta=0.4, gamma=0.2)
    
    # Initialize validators
    validators = df_agg['validator_id'].unique()
    simulator.initialize_weights(validators)
    
    test_epochs = test_df['epoch'].unique()
    test_epochs = sorted(list(test_epochs))
    
    # Escalating and then recovering CNRS for the stress-test window
    # Hardcoded to positions 33-49 in the test epoch list (middle of the test phase)
    attack_epochs = test_epochs[33:49]
    cnrs_seq = [0.70, 0.70, 0.82, 0.82, 0.91, 0.91, 0.91,  # Tier 1 buildup
                0.96,                                          # Tier 2
                0.97,                                          # Tier 3 PEAK (Self-Healing triggered)
                0.91, 0.91,                                    # Tier 1 recovery
                0.55, 0.55, 0.55, 0.55, 0.55]                 # De-escalation post self-healing
    cnrs_override = {ep: cnrs_seq[i] for i, ep in enumerate(attack_epochs)}
    print(f"[Fault Injection] Self-Healing stress-test window: epoch {attack_epochs[0]} -> {attack_epochs[-1]}")

    for epoch in test_epochs:
        epoch_data = test_df[test_df['epoch'] == epoch].copy()
        
        # Network data is same for all rows in the epoch
        network_data = epoch_data.iloc[0].copy()
        
        simulator.step_epoch(epoch, epoch_data, network_data, models,
                             force_cnrs=cnrs_override.get(epoch, None))
        
    history_df = simulator.get_history_df()
    print("Simulation complete.\n")
    
    # Print the critical self-healing window
    critical_mask = history_df['epoch'].isin(attack_epochs)
    print("=== SELF-HEALING WINDOW ===")
    print(history_df[critical_mask][['epoch','cnrs','state','qt','fsm_tier','active_committee']].to_string())
    print("===========================\n")
    print(history_df.head(5))
    print(history_df.tail(5))
    print(f"Total epochs simulated: {len(history_df)}")
    
    print("Generating Evaluation results...")
    evaluator = Evaluator(test_df, history_df, output_dir='/home/jobin/bct proj/AdaptiveConsensus/results')
    evaluator.evaluate_and_plot()

if __name__ == "__main__":
    main()
