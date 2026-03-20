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
    
    for epoch in test_epochs:
        epoch_data = test_df[test_df['epoch'] == epoch].copy()
        
        # Network data is same for all rows in the epoch
        network_data = epoch_data.iloc[0].copy()
        
        simulator.step_epoch(epoch, epoch_data, network_data, models)
        
    history_df = simulator.get_history_df()
    print("Simulation complete. History snapshot:")
    print(history_df.head(10))
    print(history_df.tail(10))
    print(f"Total epochs simulated: {len(history_df)}")
    
    print("Generating Evaluation results...")
    evaluator = Evaluator(test_df, history_df, output_dir='/home/jobin/bct proj/AdaptiveConsensus/results')
    evaluator.evaluate_and_plot()

if __name__ == "__main__":
    main()
