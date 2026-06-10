"""
Risk Scoring — Composite Network Risk Score (CNRS) and per-validator risk.
Separates risk calculation logic from the consensus simulator.
"""

import numpy as np
from config import CNRS_ALPHA, CNRS_BETA, CNRS_GAMMA


def compute_cnrs(p_fail_list, anomaly_scores, momentum,
                 alpha=None, beta=None, gamma=None):
    """
    Compute the Composite Network Risk Score (CNRS).
    CNRS = α·max(p_fail) + β·mean(anomaly) + γ·momentum
    Returns float in [0, 1].
    """
    # Use global defaults if not provided (allows dynamic override)
    a = alpha if alpha is not None else CNRS_ALPHA
    b = beta if beta is not None else CNRS_BETA
    g = gamma if gamma is not None else CNRS_GAMMA

    max_p_fail = float(np.max(p_fail_list)) if len(p_fail_list) > 0 else 0.0
    mean_anomaly = float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0
    cnrs = a * max_p_fail + b * mean_anomaly + g * momentum
    return min(max(cnrs, 0.0), 1.0)


def compute_validator_risk(p_fail, anomaly_score, is_anomaly):
    """
    Per-validator risk score combining failure probability and anomaly score.
    Returns float in [0, 1].
    """
    risk = 0.6 * p_fail + 0.4 * anomaly_score
    if is_anomaly:
        risk = min(risk + 0.2, 1.0)
    return min(max(risk, 0.0), 1.0)


def compute_validator_risks(p_fail_list, anomaly_scores, is_anomaly_list):
    """
    Compute risk scores for all validators.
    Returns array of risk scores in [0, 1].
    """
    risks = []
    for i in range(len(p_fail_list)):
        risk = compute_validator_risk(
            p_fail_list[i],
            anomaly_scores[i],
            is_anomaly_list[i]
        )
        risks.append(risk)
    return np.array(risks)
