"""
Consensus Simulator — The core adaptive consensus engine.
Implements:
  Layer 3: Risk scoring (via risk_scoring module)
  Layer 4: Adaptive reconfiguration (quorum, weights, timeout, mode)
  Layer 5: Self-healing FSM (quarantine, committee regen, safe-mode)
Simulates consensus outcomes (success/failure/fork) based on current parameters.
"""

import numpy as np
import pandas as pd
from risk_scoring import compute_cnrs, compute_validator_risks
from config import (
    QUORUM_DEFAULT, TIMEOUT_DEFAULT,
    THRESHOLD_CAUTIOUS, THRESHOLD_RESTRICTED, THRESHOLD_CRITICAL,
    QUORUM_NORMAL, QUORUM_CAUTIOUS, QUORUM_RESTRICTED, QUORUM_SAFE_MODE,
    TIER1_THRESHOLD, TIER2_THRESHOLD,
    HYSTERESIS_EPOCHS, RETRAIN_INTERVAL,
    CNRS_ALPHA, CNRS_BETA, CNRS_GAMMA,
)


class ConsensusSimulator:
    """
    Epoch-based adaptive consensus simulator with FSM self-healing.
    """

    def __init__(self, mode='adaptive'):
        """
        mode: 'static' | 'healing_only' | 'adaptive_only' | 'adaptive'
        Controls which layers are active for baseline comparison.
        """
        self.mode = mode
        self.qt = QUORUM_DEFAULT
        self.timeout = TIMEOUT_DEFAULT
        self.state = 'NORMAL'
        self.committee_size = 0
        self.weights = {}
        self.fsm_tier = 0

        # Hysteresis tracking
        self._epochs_below_critical = 0
        self._epochs_below_restricted = 0
        self._epochs_below_cautious = 0

        # Online retraining tracking
        self.epochs_since_retrain = 0
        self.rolling_data = pd.DataFrame()

        # History
        self.history = []

    def initialize_weights(self, validators):
        self.weights = {v: 1.0 for v in validators}
        self.committee_size = len(validators)

    def _simulate_consensus_outcome(self, qt, weights, timeout, validator_data,
                                     network_data, p_fail_list, anomaly_scores):
        """
        Simulate whether consensus succeeds, fails, or forks this epoch.
        Based on current parameters and validator/network state.
        Returns: (consensus_success: bool, fork_occurred: bool, consensus_time: float)
        """
        n_validators = len(validator_data)
        if n_validators == 0:
            return False, False, timeout

        # Active validators: weight > 0 and uptime == 1
        active_mask = []
        for i, (_, row) in enumerate(validator_data.iterrows()):
            vid = row['validator_id']
            w = self.weights.get(vid, 1.0)
            active_mask.append(w > 0 and row.get('uptime', 1) == 1)

        active_count = sum(active_mask)
        total_weight = sum(self.weights.get(row['validator_id'], 1.0)
                          for _, row in validator_data.iterrows())
        active_weight = sum(
            self.weights.get(row['validator_id'], 1.0)
            for i, (_, row) in enumerate(validator_data.iterrows())
            if active_mask[i]
        )

        # Quorum check: active weight / total weight >= qt
        weight_ratio = active_weight / max(total_weight, 1e-9)
        consensus_success = weight_ratio >= qt

        # Consensus time: base + latency influence + failure influence
        avg_latency = network_data.get('msg_latency_ms', 80) / 1000.0
        avg_delay = float(validator_data['vote_delay_sec'].mean()) if 'vote_delay_sec' in validator_data.columns else 0
        
        # Tradeoff: Higher qt increases safety but slightly increases consensus time (longer to collect votes)
        qt_delay = (qt - 0.67) * 2.0 
        consensus_time = avg_latency + avg_delay + qt_delay + np.random.normal(0, 0.2)
        consensus_time = max(1.0, min(consensus_time, timeout))

        if not consensus_success:
            consensus_time = timeout  # timed out

        # Fork simulation (BFT Principles):
        # A higher Quorum Threshold (qt) SIGNIFICANTLY reduces the probability of a fork (Safety)
        partition = network_data.get('partition_indicator', 0)
        mean_anom = float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0
        
        # Base probability decreases as qt increases
        base_fork_prob = max(0.01, 0.15 - (qt - 0.67)) 
        
        fork_prob = base_fork_prob
        if partition:
            fork_prob += (0.5 / qt)  # Partition is dangerous, but high qt helps
        
        # Malicious nodes increase fork risk, but high qt mitigates it
        fork_prob += (mean_anom * 0.4) / qt

        fork_occurred = np.random.random() < min(fork_prob, 0.95)
        
        # If consensus failed, a fork is very unlikely (Safety over Liveness)
        if not consensus_success:
            fork_occurred = False

        return consensus_success, fork_occurred, round(consensus_time, 3)

    def step_epoch(self, epoch, validator_data, network_data, models):
        """
        Run one epoch of the consensus loop.
        models: PredictiveModels instance (or None for static/healing-only modes).
        Returns: metrics dict for this epoch.
        """
        # Append to rolling window for potential retraining
        self.rolling_data = pd.concat([self.rolling_data, validator_data])
        min_epoch = epoch - 59
        if 'epoch' in self.rolling_data.columns:
            self.rolling_data = self.rolling_data[self.rolling_data['epoch'] >= min_epoch]

        # ─── Predictions ──────────────────────────────────────────────────
        if self.mode == 'adaptive' and models is not None and models.is_trained:
            # Full system: uses ML predictive lookahead
            p_fail_list = models.predict_failure(validator_data)
            anomaly_scores, is_anomaly = models.predict_anomaly(validator_data)
            p_fork = models.predict_fork(pd.DataFrame([network_data]))[0]
        elif self.mode in ('healing_only', 'adaptive_only'):
            # Reactive modes: use raw current-epoch signals, no ML lookahead
            p_fail_list = np.where(validator_data['uptime'].values == 0, 0.95, 0.05)
            anomaly_scores = np.where(
                validator_data['missed_vote_rate'].values > 0.4, 0.85, 0.1)
            is_anomaly = anomaly_scores > 0.7
            p_fork = 0.6 if network_data.get('partition_indicator', 0) else 0.1
        else:
            # Static: no predictions
            p_fail_list = np.zeros(len(validator_data))
            anomaly_scores = np.zeros(len(validator_data))
            is_anomaly = np.zeros(len(validator_data), dtype=bool)
            p_fork = 0.0

        # ─── Layer 3: Risk Scoring ────────────────────────────────────────
        momentum = self.history[-1]['cnrs'] if len(self.history) > 0 else 0.0
        cnrs = compute_cnrs(p_fail_list, anomaly_scores, momentum)

        validator_risks = compute_validator_risks(p_fail_list, anomaly_scores, is_anomaly)

        # ─── Layer 4: Adaptive Reconfiguration ────────────────────────────
        prev_state = self.state
        prev_tier = self.fsm_tier

        if self.mode == 'static':
            # Static: fixed parameters, no adaptation
            self.qt = QUORUM_DEFAULT
            self.timeout = TIMEOUT_DEFAULT
            self.state = 'NORMAL'
            self.fsm_tier = 0
            for v in self.weights:
                self.weights[v] = 1.0
        else:
            self._apply_adaptation(epoch, cnrs, p_fork, p_fail_list, anomaly_scores,
                                   is_anomaly, validator_data, network_data, models)

        # ─── Simulate consensus outcome ───────────────────────────────────
        consensus_success, fork_occurred, consensus_time = self._simulate_consensus_outcome(
            self.qt, self.weights, self.timeout, validator_data,
            network_data, p_fail_list, anomaly_scores)

        # ─── Build metrics ────────────────────────────────────────────────
        active_committee = sum(1 for w in self.weights.values() if w > 0)
        metrics = {
            'epoch': epoch,
            'cnrs': round(cnrs, 4),
            'p_fork': round(float(p_fork), 4),
            'max_p_fail': round(float(np.max(p_fail_list)), 4) if len(p_fail_list) > 0 else 0,
            'mean_anomaly': round(float(np.mean(anomaly_scores)), 4) if len(anomaly_scores) > 0 else 0,
            'qt': round(self.qt, 4),
            'state': self.state,
            'timeout': round(self.timeout, 2),
            'fsm_tier': self.fsm_tier,
            'active_committee': active_committee,
            'consensus_success': consensus_success,
            'fork_occurred': fork_occurred,
            'consensus_time': consensus_time,
            'state_changed': self.state != prev_state,
        }
        self.history.append(metrics)
        return metrics

    def _apply_adaptation(self, epoch, cnrs, p_fork, p_fail_list,
                           anomaly_scores, is_anomaly, validator_data,
                           network_data, models):
        """Apply Layer 4 reconfiguration and Layer 5 self-healing."""

        # ── Fork risk override ────────────────────────────────────────────
        fork_elevated = p_fork > 0.7

        # ── Determine target state based on CNRS ──────────────────────────
        if cnrs < THRESHOLD_CAUTIOUS:
            target_state = 'NORMAL'
        elif cnrs < THRESHOLD_RESTRICTED:
            target_state = 'CAUTIOUS'
        elif cnrs < THRESHOLD_CRITICAL:
            target_state = 'RESTRICTED'
        else:
            target_state = 'CRITICAL'

        # ── Hysteresis: prevent rapid oscillation ─────────────────────────
        # Only de-escalate if below threshold for HYSTERESIS_EPOCHS consecutive epochs
        if target_state == 'NORMAL' and self.state in ('CAUTIOUS', 'RESTRICTED', 'CRITICAL'):
            self._epochs_below_cautious += 1
            if self._epochs_below_cautious < HYSTERESIS_EPOCHS:
                target_state = 'CAUTIOUS'  # hold
            else:
                self._epochs_below_cautious = 0
                self._epochs_below_restricted = 0
                self._epochs_below_critical = 0
        elif target_state == 'CAUTIOUS' and self.state in ('RESTRICTED', 'CRITICAL'):
            self._epochs_below_restricted += 1
            self._epochs_below_cautious = 0
            if self._epochs_below_restricted < HYSTERESIS_EPOCHS:
                target_state = 'RESTRICTED'  # hold
            else:
                self._epochs_below_restricted = 0
                self._epochs_below_critical = 0
        elif target_state == 'RESTRICTED' and self.state == 'CRITICAL':
            self._epochs_below_critical += 1
            self._epochs_below_cautious = 0
            self._epochs_below_restricted = 0
            if self._epochs_below_critical < HYSTERESIS_EPOCHS:
                target_state = 'CRITICAL'  # hold
            else:
                self._epochs_below_critical = 0
        else:
            # Escalation or same state: reset counters
            self._epochs_below_cautious = 0
            self._epochs_below_restricted = 0
            self._epochs_below_critical = 0

        self.state = target_state

        # ── Apply state-specific parameters ───────────────────────────────
        if self.state == 'NORMAL':
            self.qt = QUORUM_NORMAL
            self.timeout = TIMEOUT_DEFAULT
            self.fsm_tier = 0
            # Restore all weights
            for v in self.weights:
                self.weights[v] = 1.0

        elif self.state == 'CAUTIOUS':
            self.qt = QUORUM_CAUTIOUS
            self.timeout = TIMEOUT_DEFAULT + 2.0
            self.fsm_tier = 0
            # Reduce weights of risky validators
            for i, (_, row) in enumerate(validator_data.iterrows()):
                vid = row['validator_id']
                if i < len(p_fail_list) and (p_fail_list[i] > 0.5 or is_anomaly[i]):
                    self.weights[vid] = 0.5
                else:
                    self.weights[vid] = max(self.weights.get(vid, 1.0), 0.5)

        elif self.state == 'RESTRICTED':
            self.qt = QUORUM_RESTRICTED
            self.timeout = TIMEOUT_DEFAULT + 4.0
            self.fsm_tier = 0
            # Isolate high-risk validators
            for i, (_, row) in enumerate(validator_data.iterrows()):
                vid = row['validator_id']
                if i < len(p_fail_list) and (p_fail_list[i] > 0.7 or anomaly_scores[i] > 0.8):
                    self.weights[vid] = 0.0
                else:
                    self.weights[vid] = max(self.weights.get(vid, 1.0), 0.3)

        elif self.state == 'CRITICAL':
            self._apply_self_healing(epoch, cnrs, p_fail_list, anomaly_scores,
                                      validator_data, models)

        # Override quorum if fork risk is independently high
        if fork_elevated and self.qt < 0.80:
            self.qt = 0.80

        # Adjust timeout based on current latency
        latency = network_data.get('msg_latency_ms', 80)
        if latency > 300:
            self.timeout = max(self.timeout, TIMEOUT_DEFAULT + 5.0)

    def _apply_self_healing(self, epoch, cnrs, p_fail_list, anomaly_scores,
                             validator_data, models):
        """Layer 5: Self-healing FSM — Tier 1/2/3 responses."""

        # Determine tier
        if cnrs < TIER1_THRESHOLD:
            self.fsm_tier = 1
        elif cnrs < TIER2_THRESHOLD:
            self.fsm_tier = 2
        else:
            self.fsm_tier = 3

        if self.fsm_tier in (1, 2):
            # Tier 1/2: Watchlist + weight reduction + optional retraining
            self.qt = QUORUM_RESTRICTED
            for i, (_, row) in enumerate(validator_data.iterrows()):
                vid = row['validator_id']
                if i < len(p_fail_list) and (p_fail_list[i] > 0.6 or anomaly_scores[i] > 0.7):
                    self.weights[vid] = 0.0  # quarantine
                else:
                    self.weights[vid] = max(self.weights.get(vid, 1.0), 0.3)

            self.epochs_since_retrain += 1
            if self.epochs_since_retrain >= RETRAIN_INTERVAL and models is not None:
                if self.mode == 'adaptive' and len(self.rolling_data) > 50:
                    models.retrain(self.rolling_data)
                    self.epochs_since_retrain = 0

        elif self.fsm_tier == 3:
            # Tier 3: Safe-mode — supermajority quorum, minimal committee
            self.qt = QUORUM_SAFE_MODE
            # Keep only the safest validators
            sorted_idx = np.argsort(p_fail_list)
            n_safe = max(4, len(sorted_idx) // 3)
            safe_vids = set(validator_data.iloc[sorted_idx[:n_safe]]['validator_id'].values)
            for v in self.weights:
                self.weights[v] = 1.0 if v in safe_vids else 0.0

    def get_history_df(self):
        return pd.DataFrame(self.history)

    def reset(self):
        """Reset simulator state for a new run."""
        self.qt = QUORUM_DEFAULT
        self.timeout = TIMEOUT_DEFAULT
        self.state = 'NORMAL'
        self.fsm_tier = 0
        self._epochs_below_critical = 0
        self._epochs_below_restricted = 0
        self._epochs_below_cautious = 0
        self.epochs_since_retrain = 0
        self.rolling_data = pd.DataFrame()
        self.history = []
        for v in self.weights:
            self.weights[v] = 1.0
