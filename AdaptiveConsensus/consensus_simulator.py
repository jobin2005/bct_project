import numpy as np
import pandas as pd

class ConsensusSimulator:
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.qt = 0.67
        self.timeout = 10.0 # baseline timeout
        self.state = 'NORMAL'
        self.committee_size = 0
        self.weights = {} # validator_id -> weight
        self.fsm_tier = 0
        self.watchlist = False
        
        # historical tracking
        self.history = []
        
        # FSM layer 5 metadata
        self.rolling_data = pd.DataFrame()
        self.epochs_since_retrain = 0
        
    def initialize_weights(self, validators):
        self.weights = {v: 1.0 for v in validators}
        self.committee_size = len(validators)

    def calculate_cnrs(self, p_fail_list, anomaly_scores, momentum):
        max_p_fail = np.max(p_fail_list) if len(p_fail_list) > 0 else 0
        mean_anomaly = np.mean(anomaly_scores) if len(anomaly_scores) > 0 else 0
        cnrs = self.alpha * max_p_fail + self.beta * mean_anomaly + self.gamma * momentum
        # Bound it strictly between 0 and 1
        return min(max(cnrs, 0.0), 1.0)
        
    def step_epoch(self, epoch, validator_data, network_data, models):
        """
        Runs the logic for a single epoch.
        """
        # Tracking rolling data for 60 epochs
        self.rolling_data = pd.concat([self.rolling_data, validator_data])
        min_rolling_epoch = epoch - 59
        self.rolling_data = self.rolling_data[self.rolling_data['epoch'] >= min_rolling_epoch]

        # 1. Predictions
        p_fail_list = models.predict_failure(validator_data)
        anomaly_scores, is_anomaly = models.predict_anomaly(validator_data)
        p_fork = models.predict_fork(pd.DataFrame([network_data]))[0]
        
        # Calculate momentum (simple proxy using previous epoch CNRS if available)
        momentum = 0.0
        if len(self.history) > 0:
            last_cnrs = self.history[-1]['cnrs']
            momentum = last_cnrs * 0.5 # exponentially decaying momentum
            
        # 2. Layer 3 Risk Scoring
        cnrs = self.calculate_cnrs(p_fail_list, anomaly_scores, momentum)
        
        # 3. Layer 4 Adaptive Reconfiguration & Layer 5 Self-Healing
        self.qt = 0.67 # Reset default
        self.state = 'NORMAL'
        self.fsm_tier = 0
        self.timeout = 10.0
        self.watchlist = False
        
        # Override for high fork risk
        if p_fork > 0.7:
            self.qt = 0.80
            self.state = 'ELEVATED_FORK_RISK'
        
        # Apply Layer 4 thresholds regardless of fork (as stated: bypass CNRS -> set qt, then GO TO LAYER 4 to apply weights)
        if cnrs < 0.30:
            if self.state != 'ELEVATED_FORK_RISK':
                self.qt = 0.67
                self.state = 'NORMAL'
            self.timeout += 2.0
            
            # Layer 5 Recovery Check (De-escalation)
            if len(self.history) > 0 and self.history[-1]['fsm_tier'] > 0:
                print(f"[Tier De-escalation] System recovered safely at epoch {epoch}.")
                self.epochs_since_retrain = 0
                for v in self.weights.keys(): 
                    self.weights[v] = 1.0 # Restore weights
                    
        elif 0.30 <= cnrs < 0.60:
            if self.state != 'ELEVATED_FORK_RISK':
                self.qt = 0.75
                self.state = 'CAUTIOUS'
            # Reduce risky validator weights
            for i, v in enumerate(validator_data['validator_id']):
                if p_fail_list[i] > 0.5 or is_anomaly[i]:
                    self.weights[v] = 0.5
                    
        elif 0.60 <= cnrs < 0.85:
            if self.state != 'ELEVATED_FORK_RISK':
                self.qt = 0.85
                self.state = 'RESTRICTED'
            # Isolate top risk
            for i, v in enumerate(validator_data['validator_id']):
                if p_fail_list[i] > 0.7 or anomaly_scores[i] > 0.8:
                    self.weights[v] = 0.0
                    
        else:
            self.state = 'CRITICAL'
            # Determine Tier based on severity
            if cnrs < 0.90:
                self.fsm_tier = 1
            elif cnrs < 0.95:
                self.fsm_tier = 2
            else:
                self.fsm_tier = 3
            
            # Layer 5 Watchlist & Retraining (Tier 1 & 2)
            if self.fsm_tier in [1, 2]:
                self.watchlist = True
                if self.state != 'ELEVATED_FORK_RISK':
                    self.qt = 0.85 # Inherit max normalized restriction
                    
                self.epochs_since_retrain += 1
                
                if self.epochs_since_retrain >= 20:
                    print(f"[Tier {self.fsm_tier} Watchlist] Triggering Online Retraining at Epoch {epoch}...")
                    models.train(self.rolling_data)
                    self.epochs_since_retrain = 0
                    
            # Layer 5 safe mode (Tier 3)
            elif self.fsm_tier == 3:
                self.qt = 0.90 # SAFE-MODE
                # Minimizing committee
                sorted_vals = np.argsort(p_fail_list)
                safe_validators = validator_data.iloc[sorted_vals[:max(4, len(sorted_vals)//3)]]['validator_id']
                for v in self.weights.keys():
                    if v not in safe_validators.values:
                        self.weights[v] = 0.0
                    else:
                        self.weights[v] = 1.0

        # Log epoch history
        metrics = {
            'epoch': epoch,
            'cnrs': cnrs,
            'p_fork': p_fork,
            'max_p_fail': np.max(p_fail_list) if len(p_fail_list) > 0 else 0,
            'mean_anomaly': np.mean(anomaly_scores) if len(anomaly_scores) > 0 else 0,
            'qt': self.qt,
            'state': self.state,
            'timeout': self.timeout,
            'fsm_tier': self.fsm_tier,
            'active_committee': sum([1 for w in self.weights.values() if w > 0])
        }
        self.history.append(metrics)
        return metrics

    def get_history_df(self):
        return pd.DataFrame(self.history)
