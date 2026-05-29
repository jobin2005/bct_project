"""
Main Experiment Runner — Predictive Self-Healing Adaptive Consensus.
Runs all baselines × scenarios × seeds, collects results, and triggers evaluation.

Closed-loop flow per epoch:
  Telemetry Generation → Feature Aggregation → Risk Prediction →
  Consensus Reconfiguration → Self-Healing (if triggered) → Consensus Outcome
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import product

from config import (
    NUM_VALIDATORS, NUM_EPOCHS, NUM_REGIONS,
    RANDOM_SEEDS, TRAIN_FRACTION, SCENARIO_PROFILES,
)
from telemetry_generator import TelemetryGenerator
from telemetry_aggregation import aggregate_telemetry
from predictive_models import PredictiveModels
from consensus_simulator import ConsensusSimulator
from scenarios import Scenario
from baselines import BASELINE_MODES
from evaluation import Evaluator


def run_single_experiment(baseline_mode, scenario_name, seed, verbose=False):
    """
    Run a single experiment: one baseline × one scenario × one seed.
    Returns: (history_df, train_accuracy_dict)
    """
    # Initialize telemetry generator
    gen = TelemetryGenerator(num_validators=NUM_VALIDATORS, num_regions=NUM_REGIONS, seed=seed)

    train_epochs = int(NUM_EPOCHS * TRAIN_FRACTION)
    test_epochs = NUM_EPOCHS - train_epochs

    validator_ids = gen.get_validator_ids()
    region_map = {v.id: v.region for v in gen.validators}

    scenario = Scenario(
        profile_name=scenario_name,
        num_test_epochs=test_epochs,
        all_validator_ids=validator_ids,
        region_map=region_map,
        seed=seed
    )

    # ─── Phase 1 & 4: Generate all raw telemetry upfront ──────────────────
    all_raw_data = []

    # Generate training epochs (normal baseline)
    for e in range(train_epochs):
        vdf, ndata = gen.generate_epoch()
        vdf_copy = vdf.copy()
        for k, v in ndata.items():
            vdf_copy[k] = v
        all_raw_data.append(vdf_copy)

    # Generate testing epochs (with scenario disturbances)
    for test_idx in range(test_epochs):
        params = scenario.get_epoch_params(test_idx)
        vdf, ndata = gen.generate_epoch(
            latency_factor=params['latency_factor'],
            load_factor=params['load_factor'],
            force_partition=params['force_partition'],
            offline_validators=params['offline_validators'],
            malicious_validators=params['malicious_validators'],
        )
        vdf_copy = vdf.copy()
        for k, v in ndata.items():
            vdf_copy[k] = v
        all_raw_data.append(vdf_copy)

    full_raw_df = pd.concat(all_raw_data, ignore_index=True)

    # ─── Phase 2: Aggregate the entire telemetry sequence once ────────────
    full_agg = aggregate_telemetry(full_raw_df)

    # Split into train and test sets
    train_agg = full_agg[full_agg['epoch'] <= train_epochs].copy()
    test_agg = full_agg[full_agg['epoch'] > train_epochs].copy()

    # ─── Phase 3: Train ML models ────────────────────────────────────────
    models = PredictiveModels(random_state=seed)
    models.train(train_agg)

    # Compute training accuracy
    train_accuracy = {}
    if models.is_trained:
        p_fail_pred = models.predict_failure(train_agg)
        y_true = (train_agg['health_label'] != 'Healthy').astype(int).values
        train_accuracy['failure_acc'] = float(np.mean((p_fail_pred > 0.5).astype(int) == y_true))

        # Fork accuracy
        epoch_df = train_agg.groupby('epoch').first().reset_index()
        p_fork_pred = models.predict_fork(epoch_df)
        if 'fork_occurrences' in epoch_df.columns:
            y_fork = (epoch_df['fork_occurrences'] > 0).astype(int).values
            train_accuracy['fork_acc'] = float(np.mean((p_fork_pred > 0.5).astype(int) == y_fork))

    # ─── Phase 5: Run closed-loop consensus simulator ────────────────────
    simulator = ConsensusSimulator(mode=baseline_mode)
    simulator.initialize_weights(validator_ids)

    test_epochs_list = sorted(test_agg['epoch'].unique())

    for e in test_epochs_list:
        current_agg = test_agg[test_agg['epoch'] == e].copy()
        if len(current_agg) == 0:
            continue

        # Network data for fork prediction
        net_row = current_agg.iloc[0].to_dict()

        # Step the consensus simulator
        metrics = simulator.step_epoch(
            epoch=e,
            validator_data=current_agg,
            network_data=net_row,
            models=models
        )

        if verbose and (e - train_epochs - 1) % 20 == 0:
            print(f"  Epoch {e}: CNRS={metrics['cnrs']:.3f} "
                  f"State={metrics['state']} Qt={metrics['qt']:.2f} "
                  f"Success={metrics['consensus_success']} Fork={metrics['fork_occurred']}")

    history_df = simulator.get_history_df()
    return history_df, train_accuracy


def main():
    """Run all experiments and generate evaluation."""
    start_time = time.time()

    # Scenarios to test
    test_scenarios = ['normal', 'outage_burst', 'latency_spike',
                      'partition', 'malicious', 'combined_stress']
    baseline_names = list(BASELINE_MODES.keys())
    seeds = RANDOM_SEEDS

    print("=" * 70)
    print("Predictive Self-Healing Adaptive Consensus — Experiment Runner")
    print("=" * 70)
    print(f"Baselines: {baseline_names}")
    print(f"Scenarios: {test_scenarios}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(baseline_names) * len(test_scenarios) * len(seeds)}")
    print("=" * 70)

    # Collect all results
    all_results = []
    all_histories = {}
    train_accuracies = []

    total = len(baseline_names) * len(test_scenarios) * len(seeds)
    count = 0

    for baseline_name in baseline_names:
        mode = BASELINE_MODES[baseline_name]['mode']
        for scenario_name in test_scenarios:
            seed_histories = []
            for seed in seeds:
                count += 1
                print(f"\n[{count}/{total}] Baseline={baseline_name} | "
                      f"Scenario={scenario_name} | Seed={seed}")

                history_df, accuracy = run_single_experiment(
                    baseline_mode=mode,
                    scenario_name=scenario_name,
                    seed=seed,
                    verbose=False
                )

                # Tag results
                history_df['baseline'] = baseline_name
                history_df['scenario'] = scenario_name
                history_df['seed'] = seed

                seed_histories.append(history_df)
                all_results.append(history_df)

                if accuracy:
                    accuracy['baseline'] = baseline_name
                    accuracy['seed'] = seed
                    train_accuracies.append(accuracy)

                print(f"  → {len(history_df)} epochs | "
                      f"Fork rate: {history_df['fork_occurred'].mean():.3f} | "
                      f"Consensus fail: {(~history_df['consensus_success']).mean():.3f}")

            key = (baseline_name, scenario_name)
            all_histories[key] = pd.concat(seed_histories, ignore_index=True)

    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    accuracy_df = pd.DataFrame(train_accuracies) if train_accuracies else pd.DataFrame()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Total result rows: {len(results_df)}")
    print(f"{'=' * 70}\n")

    # ─── Evaluation ──────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating evaluation results...")
    evaluator = Evaluator(results_df, accuracy_df, output_dir=output_dir)
    evaluator.run_full_evaluation()

    print(f"\nResults saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
