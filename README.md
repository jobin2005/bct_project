# Predictive Self-Healing Adaptive Consensus for Smart-Grid Blockchains

## 🎓 Publication-Ready Research Framework
This repository implements a **5-Layer Adaptive Consensus Pipeline** designed to solve the safety and liveness challenges of blockchain-based smart grids.

### 🚀 Key Features
*   **Predictive Risk Scoring (CNRS)**: Real-time quantification of network risk using ML-driven telemetry analysis.
*   **Self-Healing FSM**: A 3-tier finite state machine that automatically scales, stabilizes, and recovers the network during outages or partitions.
*   **High-Scale Simulation**: Validated across **50 validators** and **300+ epochs** for statistical rigor.
*   **Publication-Grade Evaluation**: Automated multi-seed benchmarking against Static (PBFT), Healing-Only, and Reactive Baselines.

### 📐 The 5-Layer Pipeline
1.  **Telemetry Generation**: Clustered smart-grid topology with parameterized disturbance injection.
2.  **Feature Engineering**: Dual-window (short/long) sliding aggregation of network vitals.
3.  **ML Risk Prediction**: Triple-model approach (RandomForest, IsolationForest, GradientBoosting) for failure and fork forecasting.
4.  **Adaptive Reconfiguration**: Dynamic adjustment of Quorum Thresholds ($qt$) and Timeouts based on risk.
5.  **Self-Healing (FSM)**: Tiered response system (Quarantine, Weighted Reconfiguration, Safe-Mode).

### 📊 Results & Performance
*   **Failure Prediction Accuracy**: **90%+**
*   **Fork Reduction**: **~6% improvement** over standard PBFT.
*   **Availability**: **~12% improvement** in consensus survival during high-stress scenarios.

### 🛠 Usage
```bash
# Run the full experiment suite
python AdaptiveConsensus/main.py

# Run Weight Sensitivity (Ablation) Study
python AdaptiveConsensus/sensitivity_analysis.py
```
*Results are saved automatically to the `AdaptiveConsensus/results/` directory.*
 simulation framework for smart-grid blockchain networks. This framework transitions the repository from disjointed, static scripts to a scientific, closed-loop telemetry-driven consensus environment that optimizes reliability, resilience, stability, security, and performance.

---

## 🚀 Key Architectural Features

The framework is built upon a **5-Layer Predictive Self-Healing Architecture**:

1. **Layer 1: Telemetry Data Collection (`telemetry_generator.py`)**
   - Simulates a smart-grid cluster topology (substations, microgrid controllers) across different geographical regions.
   - Generates live, epoch-based validator metrics (uptime, vote delays, missed votes, connectivity) and network-level telemetry (latency, variance, packet loss, partitions).

2. **Layer 2: Telemetry Aggregation (`telemetry_aggregation.py`)**
   - Employs rolling dual-window feature engineering (Short Window = 5 epochs, Long Window = 30 epochs) to capture dynamic temporal patterns.
   - Vectorized rates of change and min-max normalization ensure consistency and scale stability across feature vectors.

3. **Layer 3: Risk Scoring & Predictors (`predictive_models.py`, `risk_scoring.py`)**
   - **Validator Failure Predictor (RandomForest)**: Identifies deteriorating or faulty validator nodes.
   - **Network Anomaly Detector (IsolationForest)**: Flags structural deviations and malicious behavior.
   - **Fork Risk Predictor (Gradient Boosting)**: Forecasts upcoming block fork probabilities.
   - **Composite Network Risk Score (CNRS)**: Dynamically combines failure probability, anomaly scores, and temporal risk momentum.

4. **Layer 4: Adaptive Reconfiguration (`consensus_simulator.py`)**
   - Real-time adjustment of consensus parameters based on CNRS.
   - Adaptive quorum thresholds ($q_t$), message timeout adjustments, and individual validator voting weight reductions.

5. **Layer 5: Self-Healing FSM (`consensus_simulator.py`)**
   - Active, tiered Finite State Machine (FSM) triggered during critical risk levels.
   - Tier 1/2 responses isolate/quarantine suspect validators and trigger online retraining of ML models.
   - Tier 3 response switches to a secure-mode committee containing only highly reliable nodes.

---

## 🧪 Scenarios and Baselines

### Disturbance Scenarios (`scenarios.py`)
Disturbances are systematically injected under controlled durations:
- **Normal**: Standard grid operating environment.
- **Outage Burst**: High-voltage validator nodes drop offline.
- **Latency Spike**: Communication congestion and telemetry delays.
- **Partition**: Communication cut off between geographical clusters.
- **Malicious**: Validators exhibit Byzantine delays and voting patterns.
- **Combined Stress**: All disturbances occur in a multi-phase cascading pattern.

### Evaluated Baselines (`baselines.py`)
We evaluate and compare four distinct consensus paradigms:
- **Static Consensus**: Fixed default parameters ($q_t=0.67$) with no recovery action.
- **Healing Only**: Reactive self-healing FSM based on current metrics without predictive ML lookahead.
- **Adaptive Only**: CNRS-based quorum adjustments using current metrics without predictive ML.
- **Proposed (Full)**: The complete framework (ML Predictive Models $\rightarrow$ CNRS $\rightarrow$ Adaptive Reconfiguration $\rightarrow$ FSM Self-Healing).

---

## 🛠️ Installation and Setup

1. **Verify Python Environment**: Ensure you have Python 3.8+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r AdaptiveConsensus/requirements.txt
   ```

---

## 🏃 Running the Experiments

To run the complete 120-experiment scientific suite (4 Baselines × 6 Scenarios × 5 Random Seeds) and compile the evaluation report:

```bash
cd AdaptiveConsensus
python main.py
```

*The simulator uses highly optimized vectorized feature pipelines, completing the entire multi-seed statistical evaluation suite in under 4 minutes.*

---

## 📊 Evaluation & Metrics (`evaluation.py`)

The evaluation engine tracks **10 key performance metrics** across five academic dimensions:
- **Reliability**: Fork Rate, Consensus Failure Probability.
- **Resilience**: Mean and Max Recovery Time (epochs to return to NORMAL state).
- **Stability**: State Oscillation Rate, Quorum Threshold Variance.
- **Security**: Active Committee Size (Isolation of compromised validators).
- **Performance**: Average Consensus Time (seconds), Composite Network Risk Scores.

All metrics are compiled into a comprehensive statistical analysis with **Mean ± Standard Deviation** and **95% Confidence Intervals (CI)**.

### Generated Visualizations (Saved in `AdaptiveConsensus/results/`)
- `fig1_cnrs_comparison.png`: CNRS progression over time under combined stress.
- `fig2_comparative_bars.png`: Comparative performance bars across all baselines.
- `fig3_recovery_distribution.png`: Box plot showing recovery time distributions.
- `fig4_qt_adaptation.png`: Real-time quorum threshold adaptation.
- `fig5_committee_size.png`: Active validator committee size under self-healing eviction.
- `fig6_fork_rate_by_scenario.png`: Fork rates grouped by disturbance scenario.
- `fig7_ml_accuracy.png`: Failure and fork predictor accuracies across different random seeds.
- `evaluation_report.txt`: Comprehensive summary text report with statistics, CI tables, and percentage improvement vs. static baselines.
