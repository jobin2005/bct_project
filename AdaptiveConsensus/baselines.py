"""
Baselines — Different consensus strategies for comparative evaluation.
Each baseline is a mode string passed to ConsensusSimulator.

Modes:
  - 'static':         Fixed qt=0.67, no adaptation, no healing
  - 'healing_only':   No ML prediction, but reactive healing based on observed failures
  - 'adaptive_only':  Adaptive reconfiguration without ML prediction (uses raw metrics)
  - 'adaptive':       Full proposed system (ML prediction → risk → adapt → heal)
"""

BASELINE_MODES = {
    'static': {
        'mode': 'static',
        'label': 'Static Consensus',
        'description': 'Fixed parameters, no adaptation or healing',
        'color': '#888888',
    },
    'healing_only': {
        'mode': 'healing_only',
        'label': 'Healing Only',
        'description': 'Reactive healing based on observed failures, no ML prediction',
        'color': '#e6a817',
    },
    'adaptive_only': {
        'mode': 'adaptive_only',
        'label': 'Adaptive (No Prediction)',
        'description': 'Adaptive reconfiguration using current metrics, no ML lookahead',
        'color': '#2196F3',
    },
    'adaptive': {
        'mode': 'adaptive',
        'label': 'Proposed (Full)',
        'description': 'ML prediction → risk scoring → adaptive reconfiguration → self-healing',
        'color': '#4CAF50',
    },
}
