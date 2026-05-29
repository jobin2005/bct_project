"""
Telemetry Aggregation — Sliding window feature engineering.
Applies dual-window (short WS, long WL) aggregation to produce predictive features
from raw telemetry. Includes normalization for consistent scale across predictors.
"""

import pandas as pd
import numpy as np
from config import WINDOW_SHORT, WINDOW_LONG

WS = WINDOW_SHORT
WL = WINDOW_LONG

VALIDATOR_FEATURES = [
    'uptime', 'vote_delay_sec', 'missed_vote_rate',
    'blocks_produced', 'connectivity_degree'
]

NETWORK_FEATURES = [
    'msg_latency_ms', 'latency_variance', 'packet_loss_rate',
    'partition_indicator', 'block_finalization_time_sec',
    'quorum_margin', 'timeout_events',
    'network_stale_rate'
]


def safe_rate_of_change(short_val, long_val):
    if pd.isna(short_val) or pd.isna(long_val):
        return 0.0
    if long_val == 0:
        return short_val if short_val != 0 else 0.0
    return (short_val - long_val) / abs(long_val)


def compute_roc_series(mu_ws, mu_wl):
    denominator = mu_wl.abs()
    roc = (mu_ws - mu_wl) / denominator
    
    # where mu_wl == 0, if mu_ws != 0 then mu_ws else 0
    zero_wl = (mu_wl == 0)
    roc = roc.where(~zero_wl, mu_ws)
    
    # Handle NaNs or infinite values
    roc = roc.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return roc


def aggregate_telemetry(df):
    """
    Applies dual-window aggregation (WS, WL) to compute predictive features.
    Returns DataFrame with aggregated + normalized features.
    """
    df = df.sort_values(by=['validator_id', 'epoch']).reset_index(drop=True)

    # Ensure numeric types
    for col in VALIDATOR_FEATURES + NETWORK_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Validator-level aggregation
    val_grouped = df.groupby('validator_id')

    for feat in VALIDATOR_FEATURES:
        if feat not in df.columns:
            continue
        df[f'{feat}_mu_ws'] = val_grouped[feat].transform(
            lambda x: x.rolling(WS, min_periods=1).mean())
        df[f'{feat}_var_ws'] = val_grouped[feat].transform(
            lambda x: x.rolling(WS, min_periods=1).var().fillna(0))
        df[f'{feat}_mu_wl'] = val_grouped[feat].transform(
            lambda x: x.rolling(WL, min_periods=1).mean())
        df[f'{feat}_var_wl'] = val_grouped[feat].transform(
            lambda x: x.rolling(WL, min_periods=1).var().fillna(0))
        df[f'{feat}_roc'] = compute_roc_series(df[f'{feat}_mu_ws'], df[f'{feat}_mu_wl'])

    # 2. Network-level aggregation
    net_df = df[['epoch'] + [f for f in NETWORK_FEATURES if f in df.columns]].drop_duplicates().sort_values('epoch').reset_index(drop=True)

    for feat in NETWORK_FEATURES:
        if feat not in net_df.columns:
            continue
        net_df[f'net_{feat}_mu_ws'] = net_df[feat].rolling(WS, min_periods=1).mean()
        net_df[f'net_{feat}_var_ws'] = net_df[feat].rolling(WS, min_periods=1).var().fillna(0)
        net_df[f'net_{feat}_mu_wl'] = net_df[feat].rolling(WL, min_periods=1).mean()
        net_df[f'net_{feat}_var_wl'] = net_df[feat].rolling(WL, min_periods=1).var().fillna(0)
        net_df[f'net_{feat}_roc'] = compute_roc_series(net_df[f'net_{feat}_mu_ws'], net_df[f'net_{feat}_mu_wl'])
        net_df = net_df.rename(columns={feat: f'net_{feat}'})

    df = df.merge(net_df, on='epoch', how='left')

    return df


def normalize_features(df, feature_cols):
    """
    Min-max normalize specified feature columns to [0, 1].
    Returns (normalized_df, stats_dict) where stats_dict contains min/max per column.
    """
    stats = {}
    df_norm = df.copy()
    for col in feature_cols:
        if col not in df_norm.columns:
            continue
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        stats[col] = {'min': col_min, 'max': col_max}
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    return df_norm, stats


def apply_normalization(df, feature_cols, stats):
    """Apply previously computed normalization stats to new data."""
    df_norm = df.copy()
    for col in feature_cols:
        if col not in df_norm.columns or col not in stats:
            continue
        col_min = stats[col]['min']
        col_max = stats[col]['max']
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
            df_norm[col] = df_norm[col].clip(0, 1)
        else:
            df_norm[col] = 0.0
    return df_norm
