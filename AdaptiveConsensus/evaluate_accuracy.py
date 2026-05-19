import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from telemetry_aggregation import aggregate_telemetry
from predictive_models import PredictiveModels

try:
    # Try local first, then parent for flexibility
    import os
    data_path = 'telemetry_dataset.xlsx'
    if not os.path.exists(data_path):
        data_path = '../telemetry_dataset.xlsx'
    df = pd.read_excel(data_path)
    df_agg = aggregate_telemetry(df)
    
    split_epoch = df_agg['epoch'].median()
    train_df = df_agg[df_agg['epoch'] <= split_epoch].copy()
    test_df = df_agg[df_agg['epoch'] > split_epoch].copy()
    
    models = PredictiveModels()
    models.train(train_df)
    
    # Evaluate Failure Predictor
    features = models.extract_features(test_df)
    X_test_fail = models.scaler.transform(test_df[features].fillna(0))
    # Map "Healthy" to 0, others to 1 (Failure)
    y_test_fail = (test_df['health_label'] != 'Healthy').astype(int)
    
    y_pred_fail = models.failure_predictor.predict(X_test_fail)
    print("--- Validator Failure Predictor (Random Forest) ---")
    print("Accuracy:", round(accuracy_score(y_test_fail, y_pred_fail), 4))
    print(classification_report(y_test_fail, y_pred_fail))
    
    # Evaluate Fork Predictor
    epoch_df = test_df.groupby('epoch').first().reset_index()
    net_features = models.extract_network_features(epoch_df)
    
    # Apply standard scaling to network features in test set to fix accuracy bug
    X_test_fork = models.net_scaler.transform(epoch_df[net_features].fillna(0))
    y_test_fork = (epoch_df['fork_occurrences'] > 0).astype(int)
    
    y_pred_fork = models.fork_predictor.predict(X_test_fork)
    print("\n--- Network Fork Predictor (Gradient Boosting) ---")
    print("Accuracy:", round(accuracy_score(y_test_fork, y_pred_fork), 4))
    print(classification_report(y_test_fork, y_pred_fork))

except Exception as e:
    print("Could not evaluate:", e)
