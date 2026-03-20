import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)

class Evaluator:
    def __init__(self, raw_df, history_df, output_dir='results'):
        self.raw_df = raw_df
        self.history_df = history_df
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def evaluate_and_plot(self):
        print(f"Generating Evaluation Plots in {self.output_dir}...")
        self.plot_cnrs_vs_state()
        self.plot_qt_adaptation()
        self.plot_committee_size()
        self.plot_fork_risk_response()
        print("Done plotting.")

    def plot_cnrs_vs_state(self):
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=self.history_df, x='epoch', y='cnrs', color='purple', label='CNRS', linewidth=2)
        plt.axhline(y=0.30, color='green', linestyle='--', label='Cautious Threshold')
        plt.axhline(y=0.60, color='orange', linestyle='--', label='Restricted Threshold')
        plt.axhline(y=0.85, color='red', linestyle='--', label='Critical / Safe-Mode Threshold')
        
        plt.title('Composite Network Risk Score (CNRS) Progression Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('CNRS')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig1_cnrs_over_time.png'), dpi=300)
        plt.close()

    def plot_qt_adaptation(self):
        plt.figure(figsize=(10, 5))
        # The adaptive qt over time
        sns.lineplot(data=self.history_df, x='epoch', y='qt', color='blue', label='Adaptive Quorum (qt)', linewidth=2.5)
        # Static baseline is flat 0.67
        plt.axhline(y=0.67, color='grey', linestyle=':', label='Static Baseline (qt=0.67)', linewidth=2.5)
        
        plt.title('Dynamic Quorum Threshold Adaptation')
        plt.xlabel('Epoch')
        plt.ylabel('Quorum Threshold (qt)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig2_qt_adaptation.png'), dpi=300)
        plt.close()
        
    def plot_committee_size(self):
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=self.history_df, x='epoch', y='active_committee', color='darkorange', linewidth=2)
        
        plt.title('Active Validator Committee Size Over Epochs (Self-Healing)')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Active Validators')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig3_committee_size.png'), dpi=300)
        plt.close()

    def plot_fork_risk_response(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Fork Propability (P_fork)', color=color)
        ax1.plot(self.history_df['epoch'], self.history_df['p_fork'], color=color, alpha=0.6, label='Predicted Fork Risk')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Quorum Threshold (qt)', color=color)  
        ax2.plot(self.history_df['epoch'], self.history_df['qt'], color=color, linestyle='--', linewidth=2, label='Adaptive qt')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('System Response to Predicted Fork Risks')
        fig.tight_layout() 
        plt.savefig(os.path.join(self.output_dir, 'fig4_fork_response.png'), dpi=300)
        plt.close()
