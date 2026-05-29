"""
Predictive Models — ML-based risk prediction from telemetry features.
- Validator Failure Predictor (RandomForest)
- Anomaly Detector (IsolationForest)
- Fork Risk Predictor (GradientBoosting)
Supports online retraining and feature importance logging.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings

from config import RF_N_ESTIMATORS, RF_MAX_DEPTH, IF_CONTAMINATION, GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH

warnings.filterwarnings('ignore', category=UserWarning)


class PredictiveModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.anomaly_detector = IsolationForest(
            contamination=IF_CONTAMINATION, random_state=random_state)
        self.failure_predictor = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=random_state)
        self.fork_predictor = GradientBoostingClassifier(
            n_estimators=GB_N_ESTIMATORS, learning_rate=GB_LEARNING_RATE,
            max_depth=GB_MAX_DEPTH, subsample=0.8, random_state=random_state)
        self.scaler = StandardScaler()
        self.net_scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance_ = {}
        # Store the global min/max for anomaly score normalization
        self._anomaly_score_min = None
        self._anomaly_score_max = None

    def _validator_features(self, df):
        """Extract validator-level feature columns."""
        base = ['uptime', 'vote_delay_sec', 'missed_vote_rate', 'blocks_produced', 'connectivity_degree']
        agg = [c for c in df.columns if any(x in c for x in ['_mu_', '_var_', '_roc'])]
        existing = [f for f in base if f in df.columns]
        return existing + [c for c in agg if not c.startswith('net_')]

    def _network_features(self, df):
        """Extract network-level feature columns."""
        exclude = ['fork_occurrences', 'health_label', 'epoch']
        return [c for c in df.columns if c.startswith('net_') and not any(x in c for x in exclude)]

    def train(self, train_df):
        """Train all three models on historical telemetry data."""
        features = self._validator_features(train_df)
        net_features = self._network_features(train_df)

        if len(features) == 0 or len(train_df) == 0:
            return

        X_all = train_df[features].fillna(0).values
        self.scaler.fit(X_all)
        X_scaled = self.scaler.transform(X_all)

        # 1. Anomaly Detector — train on healthy data
        if 'health_label' in train_df.columns:
            healthy = train_df[train_df['health_label'] == 'Healthy']
            if len(healthy) > 10:
                X_h = self.scaler.transform(healthy[features].fillna(0).values)
                self.anomaly_detector.fit(X_h)
            else:
                self.anomaly_detector.fit(X_scaled)
        else:
            self.anomaly_detector.fit(X_scaled)

        # Compute global anomaly score range for stable normalization
        raw_scores = -self.anomaly_detector.score_samples(X_scaled)
        self._anomaly_score_min = float(np.percentile(raw_scores, 1))
        self._anomaly_score_max = float(np.percentile(raw_scores, 99))

        # 2. Failure Predictor
        if 'health_label' in train_df.columns:
            y_fail = (train_df['health_label'] != 'Healthy').values.astype(int)
        else:
            y_fail = np.zeros(len(X_scaled))
        if len(np.unique(y_fail)) > 1:
            self.failure_predictor.fit(X_scaled, y_fail)
            self.feature_importance_['failure'] = dict(zip(features,
                self.failure_predictor.feature_importances_))
        else:
            self.failure_predictor.fit(X_scaled, np.zeros(len(X_scaled)))

        # 3. Fork Predictor — network level, aggregate by epoch
        if len(net_features) > 0 and 'fork_occurrences' in train_df.columns:
            epoch_df = train_df.groupby('epoch').first().reset_index()
            X_net = epoch_df[net_features].fillna(0).values
            self.net_scaler.fit(X_net)
            X_net_scaled = self.net_scaler.transform(X_net)
            y_fork = (epoch_df['fork_occurrences'] > 0).astype(int)
            if len(y_fork.unique()) > 1:
                self.fork_predictor.fit(X_net_scaled, y_fork)
                self.feature_importance_['fork'] = dict(zip(net_features,
                    self.fork_predictor.feature_importances_))
            else:
                self.fork_predictor.fit(X_net_scaled, np.zeros(len(X_net_scaled)))

        self.is_trained = True

    def retrain(self, new_data):
        """Online retrain — re-fit models with new data without resetting scaler."""
        if not self.is_trained:
            self.train(new_data)
            return
        # Partial retrain: re-fit classifiers with new data using existing scaler
        features = self._validator_features(new_data)
        net_features = self._network_features(new_data)

        if len(features) == 0:
            return

        X = self.scaler.transform(new_data[features].fillna(0).values)

        if 'health_label' in new_data.columns:
            y_fail = (new_data['health_label'] != 'Healthy').values.astype(int)
            if len(np.unique(y_fail)) > 1:
                self.failure_predictor.fit(X, y_fail)

        # Retrain anomaly detector
        healthy = new_data[new_data.get('health_label', pd.Series(['Healthy'])) == 'Healthy'] if 'health_label' in new_data.columns else new_data
        if len(healthy) > 10:
            X_h = self.scaler.transform(healthy[features].fillna(0).values)
            self.anomaly_detector.fit(X_h)

    def predict_failure(self, df):
        """Returns array of failure probabilities for each validator row."""
        if not self.is_trained:
            return np.zeros(len(df))
        features = self._validator_features(df)
        X = self.scaler.transform(df[features].fillna(0).values)
        if len(self.failure_predictor.classes_) > 1:
            return self.failure_predictor.predict_proba(X)[:, 1]
        return np.zeros(len(X))

    def predict_anomaly(self, df):
        """Returns (anomaly_scores [0-1], is_anomaly boolean array)."""
        if not self.is_trained:
            return np.zeros(len(df)), np.zeros(len(df), dtype=bool)
        features = self._validator_features(df)
        X = self.scaler.transform(df[features].fillna(0).values)
        raw = -self.anomaly_detector.score_samples(X)

        # Normalize using global stats for stability
        lo = self._anomaly_score_min if self._anomaly_score_min is not None else raw.min()
        hi = self._anomaly_score_max if self._anomaly_score_max is not None else raw.max()
        if hi > lo:
            scores = np.clip((raw - lo) / (hi - lo), 0, 1)
        else:
            scores = np.zeros(len(raw))

        is_anomaly = scores > 0.7
        return scores, is_anomaly

    def predict_fork(self, network_df):
        """Returns array of fork probabilities (one per epoch-row)."""
        if not self.is_trained:
            return np.zeros(len(network_df))
        net_features = self._network_features(network_df)
        if len(net_features) == 0:
            return np.zeros(len(network_df))
        X = self.net_scaler.transform(network_df[net_features].fillna(0).values)
        if len(self.fork_predictor.classes_) > 1:
            return self.fork_predictor.predict_proba(X)[:, 1]
        return np.zeros(len(X))
