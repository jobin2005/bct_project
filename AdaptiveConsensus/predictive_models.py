import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PredictiveModels:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.failure_predictor = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.fork_predictor = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
        self.scaler = StandardScaler()
        self.net_scaler = StandardScaler()

    def extract_features(self, df):
        # Include both aggregated features and raw performance metrics for stronger signal
        base_features = ['uptime', 'vote_delay_sec', 'missed_vote_rate', 'blocks_produced', 'connectivity_degree']
        agg_features = [col for col in df.columns if any(x in col for x in ['_mu_', '_var_', '_roc'])]
        # Only include base features that actually exist in the dataframe
        existing_base = [f for f in base_features if f in df.columns]
        return existing_base + agg_features
        
    def extract_network_features(self, df):
        # Exclude the target labels from features, but keep raw network metrics (net_ prefix)
        exclude = ['fork_occurrences', 'health_label', 'epoch']
        # Also include raw metrics like net_msg_latency_ms, net_packet_loss_rate, etc.
        return [col for col in df.columns if col.startswith('net_') and not any(x in col for x in exclude)]

    def train(self, historical_df):
        features = self.extract_features(historical_df)
        net_features = self.extract_network_features(historical_df)
        
        # 1. Anomaly Detector (train on healthy data only)
        # Assuming health_label == "Healthy" is 0 in numerical form, or just filter by string
        healthy_df = historical_df[historical_df['health_label'] == 'Healthy']
        if len(healthy_df) > 0:
            X_healthy = self.scaler.fit_transform(healthy_df[features].fillna(0))
            self.anomaly_detector.fit(X_healthy)
        else:
            # Fallback if no purely healthy data
            X_all = self.scaler.fit_transform(historical_df[features].fillna(0))
            self.anomaly_detector.fit(X_all)

        # 2. Failure Predictor (predict health_label != "Healthy")
        # health_label: "Healthy" -> 0, else -> 1
        X_all = self.scaler.transform(historical_df[features].fillna(0))
        y_fail = (historical_df['health_label'] != 'Healthy').values.astype(int)
        if len(np.unique(y_fail)) > 1:
            self.failure_predictor.fit(X_all, y_fail)
        else:
            # Create a dummy model or fake data if only 1 class is present
            if len(X_all) > 0:
                self.failure_predictor.fit(X_all, np.zeros(len(X_all)))

        # 3. Fork Risk Predictor (predict fork_occurrences > 0)
        # Network level, so we aggregate by epoch first
        epoch_df = historical_df.groupby('epoch').first().reset_index()
        X_net = self.net_scaler.fit_transform(epoch_df[net_features].fillna(0))
        y_fork = (epoch_df['fork_occurrences'] > 0).astype(int)
        if len(y_fork.unique()) > 1:
            self.fork_predictor.fit(X_net, y_fork)
        else:
            if len(X_net) > 0:
                self.fork_predictor.fit(X_net, np.zeros(len(X_net)))

    def predict_anomaly(self, df):
        features = self.extract_features(df)
        X = self.scaler.transform(df[features].fillna(0))
        # IsolationForest returns 1 for inliers, -1 for outliers.
        # We want an AnomalyScore where higher is more anomalous. 
        # score_samples returns negative anomaly score (lower is more anomalous).
        # We invert it so higher is more anomalous.
        scores = -self.anomaly_detector.score_samples(X) 
        # Normalize between 0 and 1 roughly (using empirical min/max or z-score)
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        else:
            scores = np.zeros(len(scores))
        
        # Override weights logic
        threshold = 0.7 # Exceeding 70% of max anomaly score
        is_anomaly = scores > threshold
        return scores, is_anomaly

    def predict_failure(self, df):
        features = self.extract_features(df)
        X = self.scaler.transform(df[features].fillna(0))
        # returns probability of class 1 (failure)
        if len(self.failure_predictor.classes_) > 1:
            return self.failure_predictor.predict_proba(X)[:, 1]
        else:
            return np.zeros(len(X))

    def predict_fork(self, network_df):
        net_features = self.extract_network_features(network_df)
        X_net = self.net_scaler.transform(network_df[net_features].fillna(0))
        if len(self.fork_predictor.classes_) > 1:
            return self.fork_predictor.predict_proba(X_net)[:, 1]
        else:
            return np.zeros(len(X_net))
