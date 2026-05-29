"""
Scenario Framework — Parameterized disturbance injection for smart-grid simulations.
Each scenario determines what disturbances are active for a given epoch.
"""

import numpy as np
from config import SCENARIO_PROFILES


class Scenario:
    """
    Manages disturbance injection for a simulation run.
    Disturbances are injected in the middle portion of the test phase.
    """

    def __init__(self, profile_name, num_test_epochs, all_validator_ids,
                 region_map=None, seed=42):
        """
        profile_name: key from SCENARIO_PROFILES
        num_test_epochs: total number of test epochs
        all_validator_ids: list of all validator IDs
        region_map: dict mapping validator_id -> region_id (for partition scenarios)
        """
        self.profile = SCENARIO_PROFILES[profile_name]
        self.name = profile_name
        self.num_test_epochs = num_test_epochs
        self.all_validators = list(all_validator_ids)
        self.region_map = region_map or {}

        np.random.seed(seed)

        # Calculate injection window: starts at 40% into test, centered
        max_duration = max(
            self.profile.get('outage_duration', 0),
            self.profile.get('latency_duration', 0),
            self.profile.get('partition_duration', 0),
            self.profile.get('malicious_duration', 0),
            self.profile.get('load_duration', 0),
        )
        self.inject_start = int(num_test_epochs * 0.4)
        self.inject_end = self.inject_start + max(max_duration, 1)

        # Pre-select affected validators
        n_outage = int(len(self.all_validators) * self.profile.get('outage_fraction', 0))
        self.outage_validators = set(np.random.choice(
            self.all_validators, n_outage, replace=False)) if n_outage > 0 else set()

        n_malicious = int(len(self.all_validators) * self.profile.get('malicious_fraction', 0))
        remaining = [v for v in self.all_validators if v not in self.outage_validators]
        self.malicious_validators = set(np.random.choice(
            remaining, min(n_malicious, len(remaining)), replace=False)) if n_malicious > 0 else set()

        # Partition: select validators in specified regions
        self.partition_validators = set()
        for region in self.profile.get('partition_regions', []):
            self.partition_validators |= {
                v for v, r in self.region_map.items() if r == region
            }

    def get_epoch_params(self, test_epoch_index):
        """
        Returns injection parameters for a given test epoch index (0-based).
        Returns dict with keys: offline_validators, malicious_validators,
                                 latency_factor, load_factor, force_partition
        """
        result = {
            'offline_validators': set(),
            'malicious_validators': set(),
            'latency_factor': 1.0,
            'load_factor': 1.0,
            'force_partition': False,
        }

        if test_epoch_index < self.inject_start or test_epoch_index >= self.inject_end:
            return result

        # Relative position within injection window
        rel = test_epoch_index - self.inject_start

        # Outage
        if rel < self.profile.get('outage_duration', 0):
            result['offline_validators'] = self.outage_validators

        # Latency spike
        if rel < self.profile.get('latency_duration', 0):
            result['latency_factor'] = self.profile.get('latency_multiplier', 1.0)

        # Partition
        if rel < self.profile.get('partition_duration', 0) and self.partition_validators:
            result['force_partition'] = True
            result['offline_validators'] = result['offline_validators'] | self.partition_validators

        # Malicious
        if rel < self.profile.get('malicious_duration', 0):
            result['malicious_validators'] = self.malicious_validators

        # Load surge
        if rel < self.profile.get('load_duration', 0):
            result['load_factor'] = self.profile.get('load_multiplier', 1.0)

        return result

    def __repr__(self):
        return f"Scenario({self.name}: {self.profile['description']})"
