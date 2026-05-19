# Predictive Self-Healing Adaptive Consensus for Smart-Grid Blockchains

## Project Overview
This project extends the **BlockSim** framework to implement a predictive, self-healing adaptive consensus mechanism tailored for smart-grid blockchain networks. It leverages machine learning to monitor validator health and network risks (forks) in real-time, triggering autonomous defensive actions.

### 🚀 Key Features
- **Telemetry-Driven Monitoring**: Real-time collection of validator performance (vote delays, missed votes) and network health (latency, packet loss).
- **High-Accuracy Predictive Models**: 
    - **Failure Predictor**: Identifies degraded or malicious nodes with **~93% Accuracy**.
    - **Fork Predictor**: Forecasts network partition risks with **~94% Accuracy**.
- **Adaptive Consensus**: Dynamic reconfiguration based on CNRS (Consensus Network Risk Score).
- **Self-Healing Mechanisms**: Tiered responses including peer-switching, committee eviction, and partition recovery.

## Installation & Setup
Ensure you have **Python 3.8+** and the required dependencies:
```bash
pip install pandas numpy scikit-learn openpyxl xlsxwriter
```
*(Optional)* Use the provided virtual environment in `AdaptiveConsensus/venv`.

## Getting Started

### 1. Generate Telemetry Dataset
Simulate a smart-grid blockchain network with failure injections and partitions:
```bash
python generate_telemetry.py
```
This produces `telemetry_dataset.xlsx` with detailed validator and network-level metrics.

### 2. Evaluate Predictive Models
Train and verify the ML models (Random Forest & Gradient Boosting):
```bash
cd AdaptiveConsensus
python evaluate_accuracy.py
```
Outputs classification reports for both the Validator Failure and Network Fork predictors.

### 3. Run Adaptive Simulation
Run the full adaptive consensus simulator to see self-healing in action:
```bash
python AdaptiveConsensus/main.py
```

## Statistics and Results
Simulation results are saved to `telemetry_dataset.xlsx` and the `AdaptiveConsensus/results/` directory, including:
- Consensus Network Risk Scores (CNRS)
- Predictive accuracy logs
- Self-healing trigger events (Committe evictions, etc.)

## Acknowledgements
Based on the original **BlockSim** simulator. For the core simulation engine details, see the [Frontiers in Blockchain paper](https://www.frontiersin.org/articles/10.3389/fbloc.2020.00028/full).

---
*Created for the BCT Adaptive Consensus Research Project.*
