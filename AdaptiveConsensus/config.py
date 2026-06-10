"""
Centralized configuration for the Predictive Self-Healing Adaptive Consensus system.
All simulation parameters, ML hyperparameters, and scenario profiles are defined here.
"""

# ─── Simulation Parameters ────────────────────────────────────────────────────
NUM_VALIDATORS = 50
NUM_EPOCHS = 300          # per run
VOTES_PER_EPOCH = 50
BLOCK_INTERVAL = 12       # seconds between blocks
BDELAY = 6                # avg block propagation delay (seconds)

# Training uses first TRAIN_FRACTION of epochs, rest is test
TRAIN_FRACTION = 0.5

# Multi-seed runs for statistical rigor (standard for conference papers)
RANDOM_SEEDS = [42, 123, 256, 789, 1024]

# ─── Network Topology ─────────────────────────────────────────────────────────
# Smart-grid clustered topology: validators grouped into regions
NUM_REGIONS = 4
INTRA_REGION_LATENCY_MS = 30.0    # low latency within a substation cluster
INTER_REGION_LATENCY_MS = 120.0   # higher latency between regions

# ─── Consensus Defaults ───────────────────────────────────────────────────────
QUORUM_DEFAULT = 0.67             # 2/3 + 1 supermajority
TIMEOUT_DEFAULT = 10.0            # seconds
COMMITTEE_SIZE_DEFAULT = None     # None = all validators

# ─── CNRS (Composite Network Risk Score) Weights ──────────────────────────────
CNRS_ALPHA = 0.4   # weight for max validator failure probability
CNRS_BETA = 0.4    # weight for mean anomaly score
CNRS_GAMMA = 0.2   # weight for momentum (previous CNRS)

# ─── Weight Profiles (for sensitivity study) ─────────────────────────────
WEIGHT_PROFILES = [
    (0.8, 0.1, 0.1),  # Failure-heavy
    (0.4, 0.4, 0.2),  # Balanced (Default)
    (0.1, 0.8, 0.1),  # Anomaly-heavy
    (0.2, 0.2, 0.6),  # Momentum-heavy
]

# ─── FSM State Thresholds ─────────────────────────────────────────────────────
THRESHOLD_CAUTIOUS = 0.30
THRESHOLD_RESTRICTED = 0.60
THRESHOLD_CRITICAL = 0.85

# Quorum values per state
QUORUM_NORMAL = 0.67
QUORUM_CAUTIOUS = 0.75
QUORUM_RESTRICTED = 0.85
QUORUM_SAFE_MODE = 0.90

# Self-healing sub-tiers within CRITICAL
TIER1_THRESHOLD = 0.90
TIER2_THRESHOLD = 0.95

# Hysteresis: require CNRS to stay below threshold for N epochs before de-escalating
HYSTERESIS_EPOCHS = 3

# Online retraining interval (epochs in watchlist before retraining)
RETRAIN_INTERVAL = 20

# ─── Sliding Window Sizes for Feature Aggregation ────────────────────────────
WINDOW_SHORT = 5
WINDOW_LONG = 30

# ─── ML Model Hyperparameters ────────────────────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 8
IF_CONTAMINATION = 0.05
GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 4

# ─── Scenario Profiles ───────────────────────────────────────────────────────
# Each scenario is a dict specifying injection parameters.
# These are used by scenarios.py to parameterize disturbance injection.

SCENARIO_PROFILES = {
    "normal": {
        "description": "No injected disturbances — baseline network behavior",
        "outage_fraction": 0.0,
        "outage_duration": 0,
        "latency_multiplier": 1.0,
        "latency_duration": 0,
        "partition_regions": [],
        "partition_duration": 0,
        "malicious_fraction": 0.0,
        "malicious_duration": 0,
        "load_multiplier": 1.0,
        "load_duration": 0,
    },
    "outage_burst": {
        "description": "30% of validators go offline for 15 epochs",
        "outage_fraction": 0.30,
        "outage_duration": 15,
        "latency_multiplier": 1.0,
        "latency_duration": 0,
        "partition_regions": [],
        "partition_duration": 0,
        "malicious_fraction": 0.0,
        "malicious_duration": 0,
        "load_multiplier": 1.0,
        "load_duration": 0,
    },
    "latency_spike": {
        "description": "Network latency spikes 5x for 10 epochs",
        "outage_fraction": 0.0,
        "outage_duration": 0,
        "latency_multiplier": 5.0,
        "latency_duration": 10,
        "partition_regions": [],
        "partition_duration": 0,
        "malicious_fraction": 0.0,
        "malicious_duration": 0,
        "load_multiplier": 1.0,
        "load_duration": 0,
    },
    "partition": {
        "description": "Region 0 isolated from the rest for 12 epochs",
        "outage_fraction": 0.0,
        "outage_duration": 0,
        "latency_multiplier": 1.0,
        "latency_duration": 0,
        "partition_regions": [0],
        "partition_duration": 12,
        "malicious_fraction": 0.0,
        "malicious_duration": 0,
        "load_multiplier": 1.0,
        "load_duration": 0,
    },
    "malicious": {
        "description": "20% of validators send conflicting votes for 15 epochs",
        "outage_fraction": 0.0,
        "outage_duration": 0,
        "latency_multiplier": 1.0,
        "latency_duration": 0,
        "partition_regions": [],
        "partition_duration": 0,
        "malicious_fraction": 0.20,
        "malicious_duration": 15,
        "load_multiplier": 1.0,
        "load_duration": 0,
    },
    "load_surge": {
        "description": "Transaction load doubles for 10 epochs, causing congestion",
        "outage_fraction": 0.0,
        "outage_duration": 0,
        "latency_multiplier": 1.5,
        "latency_duration": 10,
        "partition_regions": [],
        "partition_duration": 0,
        "malicious_fraction": 0.0,
        "malicious_duration": 0,
        "load_multiplier": 2.0,
        "load_duration": 10,
    },
    "combined_stress": {
        "description": "Outages + latency spikes + malicious nodes simultaneously",
        "outage_fraction": 0.25,
        "outage_duration": 12,
        "latency_multiplier": 3.0,
        "latency_duration": 12,
        "partition_regions": [],
        "partition_duration": 0,
        "malicious_fraction": 0.15,
        "malicious_duration": 12,
        "load_multiplier": 1.5,
        "load_duration": 12,
    },
}

# ─── Evaluation Metrics ──────────────────────────────────────────────────────
CONFIDENCE_LEVEL = 0.95   # for confidence intervals
