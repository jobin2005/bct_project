"""
Telemetry Generator — Epoch-by-epoch validator and network telemetry.
Produces live telemetry data for the closed-loop simulation.
Supports smart-grid clustered topology and parameterized disturbance injection.
"""

import random
import math
import numpy as np
import pandas as pd
from config import (
    NUM_VALIDATORS, VOTES_PER_EPOCH, BLOCK_INTERVAL, BDELAY,
    NUM_REGIONS, INTRA_REGION_LATENCY_MS, INTER_REGION_LATENCY_MS,
)


class ValidatorNode:
    """Represents a smart-grid validator (substation, microgrid controller, etc.)."""

    def __init__(self, vid, region, num_peers):
        self.id = vid
        self.region = region
        self.hash_power = round(random.uniform(2, 15), 2)
        self.num_peers = num_peers
        # Mutable state
        self.uptime = 1
        self.vote_delay = 0.0
        self.missed_votes = 0
        self.total_votes = 0
        self.blocks_produced = 0
        self.balance = 0.0
        self.total_epochs = 0
        self.is_malicious = False
        self.is_forced_offline = False

    def simulate_epoch(self, stale_rate, latency_factor=1.0, load_factor=1.0):
        """Simulate one epoch for this validator. Returns a telemetry dict."""
        self.total_epochs += 1

        # Uptime: forced offline overrides natural failure
        if self.is_forced_offline:
            self.uptime = 0
        else:
            failure_prob = 0.03 + (self.id % 5) * 0.005
            self.uptime = 0 if random.random() < failure_prob else 1

        # Vote delay
        if self.uptime == 0:
            self.vote_delay = round(random.uniform(3.0, 10.0), 4)
        elif self.is_malicious:
            # Malicious nodes have erratic delays
            self.vote_delay = round(random.uniform(0.05, 8.0), 4)
        else:
            base = random.expovariate(1 / (0.8 * latency_factor))
            jitter = random.gauss(0, 0.2 * latency_factor)
            self.vote_delay = max(0.05, round(base + abs(jitter), 4))

        # Missed votes
        self.total_votes = VOTES_PER_EPOCH
        miss_rate_base = 0.02 if self.uptime == 1 else 0.35
        miss_rate_base += stale_rate * 0.5
        miss_rate_base += (self.vote_delay / 15.0)
        if self.is_malicious:
            miss_rate_base += random.uniform(0.1, 0.4)
        miss_rate_base *= load_factor
        miss_rate_base = min(miss_rate_base, 1.0)
        self.missed_votes = int(VOTES_PER_EPOCH * miss_rate_base + random.gauss(0, 1))
        self.missed_votes = max(0, min(self.missed_votes, VOTES_PER_EPOCH))

        # Blocks produced
        expected = (self.hash_power / 100) * 30
        self.blocks_produced = max(0, int(random.gauss(expected, math.sqrt(expected) + 0.5)))

        # Connectivity degree (dynamic ±1-2 peers)
        delta = random.choice([-2, -1, 0, 0, 0, 1, 2])
        connectivity = max(2, min(self.num_peers + delta, 50))
        if self.is_forced_offline:
            connectivity = 0

        # Reward
        reward = self.blocks_produced * 2.0
        self.balance += reward

        missed_vote_rate = round(self.missed_votes / max(1, self.total_votes), 4)

        # Health label (ground-truth for ML training)
        if random.random() < 0.10:
            health = random.choice(["Healthy", "Warning", "Degraded", "Faulty"])
        elif self.is_malicious:
            health = random.choice(["Degraded", "Faulty"])
        elif self.uptime == 0:
            health = "Faulty"
        elif self.vote_delay > 3.0 or missed_vote_rate > 0.25:
            health = "Degraded"
        elif self.vote_delay > 1.5 or missed_vote_rate > 0.10:
            health = "Warning"
        else:
            health = "Healthy"

        return {
            "validator_id": self.id,
            "region": self.region,
            "hash_power_pct": self.hash_power,
            "uptime": self.uptime,
            "vote_delay_sec": self.vote_delay,
            "missed_votes": self.missed_votes,
            "total_votes": self.total_votes,
            "missed_vote_rate": missed_vote_rate,
            "blocks_produced": self.blocks_produced,
            "connectivity_degree": connectivity,
            "health_label": health,
        }


class NetworkState:
    """Tracks network-level state across epochs."""

    def __init__(self):
        self.partition_active = False
        self.partition_countdown = 0

    def simulate_epoch(self, latency_factor=1.0, force_partition=False):
        """Produce network-level and consensus-level telemetry for one epoch."""
        # Partition logic
        if force_partition:
            self.partition_active = True
            self.partition_countdown = max(self.partition_countdown, 1)
        elif not self.partition_active:
            if random.random() < 0.03:
                self.partition_active = True
                self.partition_countdown = random.randint(3, 8)
        else:
            self.partition_countdown -= 1
            if self.partition_countdown <= 0:
                self.partition_active = False

        # Network telemetry
        if self.partition_active:
            msg_latency = round(random.uniform(400, 1200) * latency_factor, 2)
            latency_var = round(random.uniform(150, 400) * latency_factor, 2)
            packet_loss = round(random.uniform(0.15, 0.60), 4)
            partition_flag = 1
        else:
            msg_latency = round(max(10, random.lognormvariate(math.log(80), 0.5)) * latency_factor, 2)
            latency_var = round(random.uniform(5, msg_latency * 0.4), 2)
            packet_loss = round(max(0, random.gauss(0.02 * latency_factor, 0.015)), 4)
            partition_flag = 0

        # Stale rate
        total_blocks = random.randint(25, 35)
        stale_blocks = random.randint(0, max(1, int(total_blocks * 0.15)))
        stale_rate = stale_blocks / max(1, total_blocks)

        # Consensus telemetry
        lat_factor = msg_latency / 100.0
        base_finalization = BLOCK_INTERVAL * lat_factor
        if self.partition_active:
            base_finalization *= random.uniform(3.0, 8.0)
        loss_penalty = packet_loss * 20
        finalization_time = round(
            max(BLOCK_INTERVAL, base_finalization + loss_penalty + random.gauss(0, 1.5)), 3
        )

        if self.partition_active:
            quorum_margin = round(max(0.0, random.gauss(0.05, 0.08)), 4)
        elif stale_rate > 0.10:
            quorum_margin = round(max(0.0, random.gauss(0.30, 0.10)), 4)
        else:
            quorum_margin = round(min(1.0, max(0.0, random.gauss(0.82, 0.10))), 4)

        base_timeout_prob = 0.05 + packet_loss * 0.8 + (0.5 if self.partition_active else 0)
        timeout_events = sum(1 for _ in range(10) if random.random() < min(base_timeout_prob, 0.95))

        fork_prob = 0.05 + stale_rate * 2.0 + (0.50 if self.partition_active else 0)
        is_fork = fork_prob > 0.45
        if random.random() < 0.06:
            is_fork = not is_fork
        fork_occurrences = random.randint(1, 3) if is_fork else 0

        return {
            "msg_latency_ms": msg_latency,
            "latency_variance": latency_var,
            "packet_loss_rate": min(packet_loss, 1.0),
            "partition_indicator": partition_flag,
            "block_finalization_time_sec": finalization_time,
            "quorum_margin": quorum_margin,
            "timeout_events": timeout_events,
            "fork_occurrences": fork_occurrences,
            "network_stale_rate": round(stale_rate, 4),
        }, stale_rate


class TelemetryGenerator:
    """
    Generates telemetry epoch-by-epoch for the closed-loop simulation.
    Validators are arranged in a smart-grid clustered topology.
    """

    def __init__(self, num_validators=NUM_VALIDATORS, num_regions=NUM_REGIONS, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.num_validators = num_validators
        self.num_regions = num_regions

        # Assign validators to regions (clusters)
        validators_per_region = num_validators // num_regions
        self.validators = []
        for i in range(num_validators):
            region = min(i // validators_per_region, num_regions - 1)
            peers = random.randint(4, 20)
            self.validators.append(ValidatorNode(vid=i, region=region, num_peers=peers))

        self.network = NetworkState()
        self.epoch = 0

    def generate_epoch(self, latency_factor=1.0, load_factor=1.0,
                       force_partition=False, offline_validators=None,
                       malicious_validators=None):
        """
        Generate telemetry for a single epoch.
        Returns: (validator_df, network_data_dict)
        """
        self.epoch += 1

        # Apply forced states
        for v in self.validators:
            v.is_forced_offline = (offline_validators is not None and v.id in offline_validators)
            v.is_malicious = (malicious_validators is not None and v.id in malicious_validators)

        # Network telemetry
        net_data, stale_rate = self.network.simulate_epoch(
            latency_factor=latency_factor,
            force_partition=force_partition
        )

        # Validator telemetry
        rows = []
        for v in self.validators:
            row = v.simulate_epoch(stale_rate, latency_factor, load_factor)
            row["epoch"] = self.epoch
            row.update(net_data)
            rows.append(row)

        validator_df = pd.DataFrame(rows)
        return validator_df, net_data

    def get_validator_ids(self):
        return [v.id for v in self.validators]

    def get_region_validators(self, region_id):
        return [v.id for v in self.validators if v.region == region_id]

    def reset(self, seed=42):
        """Reset the generator for a new run."""
        random.seed(seed)
        np.random.seed(seed)
        for v in self.validators:
            v.uptime = 1
            v.vote_delay = 0.0
            v.missed_votes = 0
            v.total_votes = 0
            v.blocks_produced = 0
            v.balance = 0.0
            v.total_epochs = 0
            v.is_malicious = False
            v.is_forced_offline = False
        self.network = NetworkState()
        self.epoch = 0
