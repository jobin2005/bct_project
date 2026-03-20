import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PredictiveModels:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.failure_predictor = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.fork_predictor = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()

    def extract_features(self, df):
        # We extract the '_mu_ws', '_mu_wl', '_var_ws', '_var_wl', '_roc' features
        features = [col for col in df.columns if any(x in col for x in ['_mu_', '_var_', '_roc'])]
        return features
        
    def extract_network_features(self, df):
        features = [col for col in df.columns if 'net_' in col and any(x in col for x in ['_mu_', '_var_', '_roc'])]
        return features

    def train(self, historical_df):
        features = self.extract_features(historical_df)
        net_features = self.extract_network_features(historical_df)
        
        # 1. Anomaly Detector (train on healthy data only)
        # assuming health_label == 1 means healthy
        healthy_df = historical_df[historical_df['health_label'] == 1]
        if len(healthy_df) > 0:
            X_healthy = self.scaler.fit_transform(healthy_df[features].fillna(0))
            self.anomaly_detector.fit(X_healthy)
        else:
            # Fallback if no purely healthy data
            X_all = self.scaler.fit_transform(historical_df[features].fillna(0))
            self.anomaly_detector.fit(X_all)

        # 2. Failure Predictor (predict health_label == 0)
        # health_label: 1=healthy, 0=failed/malicious
        X_all = self.scaler.transform(historical_df[features].fillna(0))
        y_fail = (historical_df['health_label'] == 0).astype(int)
        if len(y_fail.unique()) > 1:
            self.failure_predictor.fit(X_all, y_fail)
        else:
            # Create a dummy model or fake data if only 1 class is present
            if len(X_all) > 0:
                self.failure_predictor.fit(X_all, np.zeros(len(X_all)))

        # 3. Fork Risk Predictor (predict fork_occurrences > 0)
        # Network level, so we aggregate by epoch first
        epoch_df = historical_df.groupby('epoch').first().reset_index()
        X_net = epoch_df[net_features].fillna(0)
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
        X_net = network_df[net_features].fillna(0)
        if len(self.fork_predictor.classes_) > 1:
            return self.fork_predictor.predict_proba(X_net)[:, 1]
        else:
            return np.zeros(len(X_net))
