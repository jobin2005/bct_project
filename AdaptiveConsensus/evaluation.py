"""
Evaluation Module — Comprehensive metrics, statistical analysis, and publication-quality plots.
Computes reliability, resilience, stability, security, and adaptation metrics.
Generates comparative plots, tables with confidence intervals, and detailed analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as scipy_stats

from config import CONFIDENCE_LEVEL, RANDOM_SEEDS
from baselines import BASELINE_MODES

sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'


class Evaluator:
    """Comprehensive evaluation of consensus simulation results."""

    def __init__(self, results_df, accuracy_df, output_dir='results'):
        """
        results_df: Combined DataFrame from all experiments (with baseline, scenario, seed columns)
        accuracy_df: Training accuracy for ML models
        """
        self.df = results_df
        self.accuracy_df = accuracy_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_full_evaluation(self):
        """Run all evaluation steps."""
        print("  [1/6] Computing metrics...")
        metrics_df = self.compute_metrics()
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_summary.csv'), index=False)

        print("  [2/6] Computing statistical comparison table...")
        stats_table = self.compute_statistical_table(metrics_df)
        stats_table.to_csv(os.path.join(self.output_dir, 'statistical_comparison.csv'), index=False)

        print("  [3/6] Plotting CNRS and state transitions...")
        self.plot_cnrs_comparison()

        print("  [4/6] Plotting comparative bar charts...")
        self.plot_comparative_bars(metrics_df)

        print("  [5/6] Plotting detailed analysis plots...")
        self.plot_recovery_time_distribution(metrics_df)
        self.plot_qt_adaptation_comparison()
        self.plot_committee_size_comparison()
        self.plot_fork_rate_by_scenario(metrics_df)

        print("  [6/6] Plotting ML model performance...")
        self.plot_ml_accuracy()

        # Save the full metrics table as formatted text
        self._save_summary_report(metrics_df, stats_table)
        print("  Evaluation complete.")

    # ─── Metric Computation ──────────────────────────────────────────────

    def compute_metrics(self):
        """Compute all evaluation metrics per (baseline, scenario, seed)."""
        groups = self.df.groupby(['baseline', 'scenario', 'seed'])
        rows = []

        for (baseline, scenario, seed), gdf in groups:
            n_epochs = len(gdf)
            if n_epochs == 0:
                continue

            # Reliability
            fork_rate = gdf['fork_occurred'].mean()
            consensus_failure_prob = (~gdf['consensus_success']).mean()

            # Resilience — Recovery time: epochs from CRITICAL back to NORMAL
            recovery_times = self._compute_recovery_times(gdf)
            mean_recovery = np.mean(recovery_times) if recovery_times else 0
            max_recovery = np.max(recovery_times) if recovery_times else 0

            # Stability
            state_changes = gdf['state_changed'].sum()
            state_oscillation_rate = state_changes / max(n_epochs, 1)
            qt_variance = gdf['qt'].var()

            # Adaptation
            reconfig_frequency = state_changes / max(n_epochs, 1)

            # Average consensus time
            avg_consensus_time = gdf['consensus_time'].mean()

            # CNRS stats
            mean_cnrs = gdf['cnrs'].mean()
            max_cnrs = gdf['cnrs'].max()

            # Active committee
            min_committee = gdf['active_committee'].min()
            mean_committee = gdf['active_committee'].mean()

            rows.append({
                'baseline': baseline,
                'scenario': scenario,
                'seed': seed,
                'fork_rate': round(fork_rate, 4),
                'consensus_failure_prob': round(consensus_failure_prob, 4),
                'mean_recovery_time': round(mean_recovery, 2),
                'max_recovery_time': max_recovery,
                'state_oscillation_rate': round(state_oscillation_rate, 4),
                'qt_variance': round(qt_variance, 6),
                'reconfig_frequency': round(reconfig_frequency, 4),
                'avg_consensus_time': round(avg_consensus_time, 3),
                'mean_cnrs': round(mean_cnrs, 4),
                'max_cnrs': round(max_cnrs, 4),
                'min_committee': min_committee,
                'mean_committee': round(mean_committee, 1),
                'n_epochs': n_epochs,
            })

        return pd.DataFrame(rows)

    def _compute_recovery_times(self, gdf):
        """Compute recovery times: epochs from entering CRITICAL to exiting."""
        recovery_times = []
        in_critical = False
        critical_start = 0

        for _, row in gdf.iterrows():
            if row['state'] == 'CRITICAL' and not in_critical:
                in_critical = True
                critical_start = row['epoch']
            elif row['state'] != 'CRITICAL' and in_critical:
                in_critical = False
                recovery_times.append(row['epoch'] - critical_start)

        return recovery_times

    # ─── Statistical Table ───────────────────────────────────────────────

    def compute_statistical_table(self, metrics_df):
        """
        Compute mean ± std and 95% CI for each metric, grouped by baseline × scenario.
        """
        metric_cols = ['fork_rate', 'consensus_failure_prob', 'mean_recovery_time',
                       'state_oscillation_rate', 'avg_consensus_time', 'mean_cnrs']

        rows = []
        for (baseline, scenario), gdf in metrics_df.groupby(['baseline', 'scenario']):
            row = {'baseline': baseline, 'scenario': scenario, 'n_seeds': len(gdf)}
            for col in metric_cols:
                values = gdf[col].values
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                n = len(values)
                # 95% CI
                if n > 1:
                    ci = scipy_stats.t.interval(CONFIDENCE_LEVEL, df=n - 1,
                                                 loc=mean, scale=std / np.sqrt(n))
                    ci_low, ci_high = ci
                else:
                    ci_low, ci_high = mean, mean

                row[f'{col}_mean'] = round(mean, 4)
                row[f'{col}_std'] = round(std, 4)
                row[f'{col}_ci_low'] = round(ci_low, 4)
                row[f'{col}_ci_high'] = round(ci_high, 4)
            rows.append(row)

        return pd.DataFrame(rows)

    # ─── Plotting ────────────────────────────────────────────────────────

    def plot_cnrs_comparison(self):
        """Plot CNRS over epochs for each baseline under combined_stress scenario."""
        scenario = 'combined_stress'
        fig, ax = plt.subplots(figsize=(12, 5))

        for bl_name, bl_info in BASELINE_MODES.items():
            mask = (self.df['baseline'] == bl_name) & (self.df['scenario'] == scenario)
            subset = self.df[mask]
            if len(subset) == 0:
                continue
            # Average across seeds
            avg = subset.groupby('epoch')['cnrs'].mean().reset_index()
            ax.plot(avg['epoch'], avg['cnrs'], label=bl_info['label'],
                    color=bl_info['color'], linewidth=2)

        ax.axhline(y=0.30, color='green', linestyle='--', alpha=0.5, label='Cautious')
        ax.axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label='Restricted')
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Critical')

        ax.set_title('CNRS Progression Under Combined Stress', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CNRS')
        ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig1_cnrs_comparison.png'))
        plt.close()

    def plot_comparative_bars(self, metrics_df):
        """Bar chart comparing all baselines across key metrics (averaged over scenarios and seeds)."""
        metric_cols = ['fork_rate', 'consensus_failure_prob', 'mean_recovery_time',
                       'avg_consensus_time']
        metric_labels = ['Fork Rate', 'Consensus Failure\nProbability',
                         'Mean Recovery\nTime (epochs)', 'Avg Consensus\nTime (s)']

        fig, axes = plt.subplots(1, len(metric_cols), figsize=(16, 5))

        for idx, (col, label) in enumerate(zip(metric_cols, metric_labels)):
            ax = axes[idx]
            bl_names = list(BASELINE_MODES.keys())
            means = []
            stds = []
            colors = []

            for bl in bl_names:
                vals = metrics_df[metrics_df['baseline'] == bl][col].values
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                colors.append(BASELINE_MODES[bl]['color'])

            x = np.arange(len(bl_names))
            bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([BASELINE_MODES[b]['label'] for b in bl_names],
                               rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(label, fontsize=10, fontweight='bold')

        plt.suptitle('Comparative Performance Across Baselines (All Scenarios)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig2_comparative_bars.png'))
        plt.close()

    def plot_recovery_time_distribution(self, metrics_df):
        """Box plot of recovery times across baselines."""
        fig, ax = plt.subplots(figsize=(10, 5))

        data_to_plot = []
        labels = []
        colors = []
        for bl_name in BASELINE_MODES:
            vals = metrics_df[metrics_df['baseline'] == bl_name]['mean_recovery_time'].values
            data_to_plot.append(vals)
            labels.append(BASELINE_MODES[bl_name]['label'])
            colors.append(BASELINE_MODES[bl_name]['color'])

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title('Recovery Time Distribution Across Baselines', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Recovery Time (epochs)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig3_recovery_distribution.png'))
        plt.close()

    def plot_qt_adaptation_comparison(self):
        """Quorum threshold over time for each baseline under combined_stress."""
        scenario = 'combined_stress'
        fig, ax = plt.subplots(figsize=(12, 5))

        for bl_name, bl_info in BASELINE_MODES.items():
            mask = (self.df['baseline'] == bl_name) & (self.df['scenario'] == scenario)
            subset = self.df[mask]
            if len(subset) == 0:
                continue
            avg = subset.groupby('epoch')['qt'].mean().reset_index()
            ax.plot(avg['epoch'], avg['qt'], label=bl_info['label'],
                    color=bl_info['color'], linewidth=2)

        ax.axhline(y=0.67, color='grey', linestyle=':', alpha=0.5, label='Default (0.67)')
        ax.set_title('Dynamic Quorum Threshold Adaptation Under Combined Stress',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Quorum Threshold (qt)')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig4_qt_adaptation.png'))
        plt.close()

    def plot_committee_size_comparison(self):
        """Active committee size over time."""
        scenario = 'combined_stress'
        fig, ax = plt.subplots(figsize=(12, 5))

        for bl_name, bl_info in BASELINE_MODES.items():
            mask = (self.df['baseline'] == bl_name) & (self.df['scenario'] == scenario)
            subset = self.df[mask]
            if len(subset) == 0:
                continue
            avg = subset.groupby('epoch')['active_committee'].mean().reset_index()
            ax.plot(avg['epoch'], avg['active_committee'], label=bl_info['label'],
                    color=bl_info['color'], linewidth=2)

        ax.set_title('Active Validator Committee Size (Self-Healing Response)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Active Validators')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig5_committee_size.png'))
        plt.close()

    def plot_fork_rate_by_scenario(self, metrics_df):
        """Grouped bar chart: fork rate per scenario per baseline."""
        scenarios = metrics_df['scenario'].unique()
        bl_names = list(BASELINE_MODES.keys())

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(scenarios))
        width = 0.18

        for i, bl in enumerate(bl_names):
            means = []
            stds = []
            for sc in scenarios:
                vals = metrics_df[(metrics_df['baseline'] == bl) & (metrics_df['scenario'] == sc)]['fork_rate'].values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)

            offset = (i - len(bl_names) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=BASELINE_MODES[bl]['label'],
                   color=BASELINE_MODES[bl]['color'], alpha=0.85,
                   edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha='right')
        ax.set_ylabel('Fork Rate')
        ax.set_title('Fork Rate by Scenario and Baseline', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig6_fork_rate_by_scenario.png'))
        plt.close()

    def plot_ml_accuracy(self):
        """Plot ML model training accuracy across seeds."""
        if self.accuracy_df is None or len(self.accuracy_df) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Only plot for adaptive baseline
        adaptive_acc = self.accuracy_df[self.accuracy_df['baseline'] == 'adaptive']

        if 'failure_acc' in adaptive_acc.columns and len(adaptive_acc) > 0:
            ax = axes[0]
            ax.bar(range(len(adaptive_acc)), adaptive_acc['failure_acc'].values,
                   color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Seed')
            ax.set_ylabel('Accuracy')
            ax.set_title('Failure Predictor Accuracy', fontweight='bold')
            ax.set_ylim(0, 1)
            mean_acc = adaptive_acc['failure_acc'].mean()
            ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
            ax.legend()

        if 'fork_acc' in adaptive_acc.columns and len(adaptive_acc) > 0:
            ax = axes[1]
            ax.bar(range(len(adaptive_acc)), adaptive_acc['fork_acc'].values,
                   color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Seed')
            ax.set_ylabel('Accuracy')
            ax.set_title('Fork Predictor Accuracy', fontweight='bold')
            ax.set_ylim(0, 1)
            mean_acc = adaptive_acc['fork_acc'].mean()
            ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
            ax.legend()

        plt.suptitle('ML Model Performance Across Seeds', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig7_ml_accuracy.png'))
        plt.close()

    # ─── Summary Report ──────────────────────────────────────────────────

    def _save_summary_report(self, metrics_df, stats_table):
        """Save a text summary report."""
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PREDICTIVE SELF-HEALING ADAPTIVE CONSENSUS — EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Overall summary per baseline
            f.write("OVERALL PERFORMANCE SUMMARY (averaged across all scenarios and seeds)\n")
            f.write("-" * 70 + "\n")
            for bl in BASELINE_MODES:
                bl_data = metrics_df[metrics_df['baseline'] == bl]
                if len(bl_data) == 0:
                    continue
                f.write(f"\n{BASELINE_MODES[bl]['label']}:\n")
                f.write(f"  Fork Rate:              {bl_data['fork_rate'].mean():.4f} ± {bl_data['fork_rate'].std():.4f}\n")
                f.write(f"  Consensus Failure:      {bl_data['consensus_failure_prob'].mean():.4f} ± {bl_data['consensus_failure_prob'].std():.4f}\n")
                f.write(f"  Mean Recovery Time:     {bl_data['mean_recovery_time'].mean():.2f} ± {bl_data['mean_recovery_time'].std():.2f}\n")
                f.write(f"  State Oscillation Rate: {bl_data['state_oscillation_rate'].mean():.4f} ± {bl_data['state_oscillation_rate'].std():.4f}\n")
                f.write(f"  Avg Consensus Time:     {bl_data['avg_consensus_time'].mean():.3f} ± {bl_data['avg_consensus_time'].std():.3f}\n")
                f.write(f"  Mean CNRS:              {bl_data['mean_cnrs'].mean():.4f} ± {bl_data['mean_cnrs'].std():.4f}\n")

            # Improvement table
            f.write(f"\n\n{'=' * 70}\n")
            f.write("IMPROVEMENT OF PROPOSED SYSTEM vs STATIC BASELINE\n")
            f.write("-" * 70 + "\n")

            static = metrics_df[metrics_df['baseline'] == 'static']
            proposed = metrics_df[metrics_df['baseline'] == 'adaptive']

            if len(static) > 0 and len(proposed) > 0:
                for col, label in [('fork_rate', 'Fork Rate'),
                                   ('consensus_failure_prob', 'Consensus Failure'),
                                   ('mean_recovery_time', 'Recovery Time'),
                                   ('avg_consensus_time', 'Consensus Time')]:
                    s_mean = static[col].mean()
                    p_mean = proposed[col].mean()
                    if s_mean > 0:
                        improvement = ((s_mean - p_mean) / s_mean) * 100
                        f.write(f"  {label:30s}: {improvement:+.1f}% improvement\n")

            # Statistical comparison table
            f.write(f"\n\n{'=' * 70}\n")
            f.write("DETAILED STATISTICAL COMPARISON (mean ± std, 95% CI)\n")
            f.write("-" * 70 + "\n")
            f.write(stats_table.to_string(index=False))
            f.write("\n")

            # ─── ML Model Performance ──────────────────────────────────────
            if self.accuracy_df is not None and len(self.accuracy_df) > 0:
                f.write(f"\n\n{'=' * 70}\n")
                f.write("ML MODEL PERFORMANCE (Cross-Seed Validation)\n")
                f.write("-" * 70 + "\n")
                
                adaptive_acc = self.accuracy_df[self.accuracy_df['baseline'] == 'adaptive']
                if len(adaptive_acc) > 0:
                    if 'failure_acc' in adaptive_acc.columns:
                        f.write(f"  Failure Predictor Accuracy : {adaptive_acc['failure_acc'].mean():.4f} "
                               f"± {adaptive_acc['failure_acc'].std():.4f}\n")
                    if 'fork_acc' in adaptive_acc.columns:
                        f.write(f"  Fork Predictor Accuracy    : {adaptive_acc['fork_acc'].mean():.4f} "
                               f"± {adaptive_acc['fork_acc'].std():.4f}\n")
                f.write("=" * 70 + "\n")

        print(f"  Report saved to {report_path}")
