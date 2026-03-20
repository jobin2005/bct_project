import pandas as pd
import numpy as np

WS = 5
WL = 30

VALIDATOR_FEATURES = [
    'uptime', 'vote_delay_sec', 'missed_vote_rate', 
    'blocks_produced', 'connectivity_degree'
]

NETWORK_FEATURES = [
    'msg_latency_ms', 'latency_variance', 'packet_loss_rate',
    'partition_indicator', 'block_finalization_time_sec',
    'quorum_margin', 'timeout_events', 'fork_occurrences'
]

def safe_rate_of_change(short_val, long_val):
    if pd.isna(short_val) or pd.isna(long_val):
        return 0.0
    if long_val == 0:
        return short_val if short_val != 0 else 0.0
    return (short_val - long_val) / abs(long_val)

def aggregate_telemetry(df):
    """
    Applies dual-window aggregation (WS=5, WL=30) to compute predictive features.
    """
    df = df.sort_values(by=['validator_id', 'epoch']).reset_index(drop=True)
    
    # 1. Validator-level aggregation
    val_grouped = df.groupby('validator_id')
    
    for feat in VALIDATOR_FEATURES:
        # Short window (mean, var)
        df[f'{feat}_mu_ws'] = val_grouped[feat].transform(lambda x: x.rolling(WS, min_periods=1).mean())
        df[f'{feat}_var_ws'] = val_grouped[feat].transform(lambda x: x.rolling(WS, min_periods=1).var().fillna(0))
        
        # Long window (mean, var)
        df[f'{feat}_mu_wl'] = val_grouped[feat].transform(lambda x: x.rolling(WL, min_periods=1).mean())
        df[f'{feat}_var_wl'] = val_grouped[feat].transform(lambda x: x.rolling(WL, min_periods=1).var().fillna(0))
        
        # Rate of change
        df[f'{feat}_roc'] = df.apply(lambda row: safe_rate_of_change(row[f'{feat}_mu_ws'], row[f'{feat}_mu_wl']), axis=1)

    # 2. Network-level aggregation
    # Since network features are identical for all validators in the same epoch, we can just aggregate over epochs directly,
    # or just assume the dataframe's epoch sequential order works if we group by epoch first.
    # To be safe, let's extract unique network stats per epoch, calculate rolling there, then merge back.
    
    net_df = df[['epoch'] + NETWORK_FEATURES].drop_duplicates().sort_values('epoch').reset_index(drop=True)
    
    for feat in NETWORK_FEATURES:
        net_df[f'net_{feat}_mu_ws'] = net_df[feat].rolling(WS, min_periods=1).mean()
        net_df[f'net_{feat}_var_ws'] = net_df[feat].rolling(WS, min_periods=1).var().fillna(0)
        
        net_df[f'net_{feat}_mu_wl'] = net_df[feat].rolling(WL, min_periods=1).mean()
        net_df[f'net_{feat}_var_wl'] = net_df[feat].rolling(WL, min_periods=1).var().fillna(0)
        
        net_df[f'net_{feat}_roc'] = net_df.apply(lambda row: safe_rate_of_change(row[f'net_{feat}_mu_ws'], row[f'net_{feat}_mu_wl']), axis=1)
        
        # Drop original feature from net_df to avoid collision on merge
        net_df = net_df.drop(columns=[feat])
        
    df = df.merge(net_df, on='epoch', how='left')
    
    return df
