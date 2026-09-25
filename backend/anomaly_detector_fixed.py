import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import xgboost as xgb

# --- Fix for Plotting Errors in Headless Environments ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Error initializing matplotlib: {e}", file=sys.stderr)

# --- Configuration & Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, '..', 'data', 'wind_solar.csv')
CONTAMINATION_RATE = 0.0001  # 0.01% contamination for very sparse anomalies

PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

COMPARISON_METRICS = {
    'FFNN': {'Accuracy': 0.978, 'Precision': 0.042, 'Recall': 0.006, 'F1 Score': 0.010, 'AUC': 0.490},
    'LSTM': {'Accuracy': 0.982, 'Precision': 0.800, 'Recall': 0.048, 'F1 Score': 0.090, 'AUC': 0.471},
    'Meta-Ensemble (XGBoost)': {'Accuracy': 0.981, 'Precision': 1.000, 'Recall': 0.006, 'F1 Score': 0.012, 'AUC': 0.858},
}

def clean_and_feature_engineer(df):
    """Clean data and add features as in the notebook."""
    df = df[df['Source'] == 'Solar'].copy()
    mean_prod = df['Production'].mean()
    df['Production'] = df['Production'].fillna(mean_prod)
    df['hour'] = pd.to_datetime(df['StartHour'].astype(str).str.replace('24:00:00', '00:00:00'), format='%H:%M:%S', errors='coerce').dt.hour
    df['hour'] = df['hour'].fillna(df['hour'].mode()[0])
    df['monthName'] = df['monthName'].astype(str)
    df['month'] = pd.to_datetime(df['monthName'], format='%B', errors='coerce').dt.month
    df['month'] = df['month'].fillna(df['month'].mode()[0])
    df['dayName'] = df['dayName'].astype(str)
    df['dayofweek'] = pd.to_datetime(df['dayName'], format='%A', errors='coerce').dt.dayofweek
    df['dayofweek'] = df['dayofweek'].fillna(df['dayofweek'].mode()[0]).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    for w in [3, 6, 24]:
        df[f'roll_mean_{w}'] = df['Production'].rolling(w).mean().shift(1)
        df[f'roll_std_{w}'] = df['Production'].rolling(w).std().shift(1)
    df = df.ffill().fillna(0)
    return df

def inject_anomalies(df, frac=0.02, seed=42):
    """Inject synthetic anomalies as in the notebook."""
    np.random.seed(seed)
    df = df.copy()
    labels = np.zeros(len(df))
    idxs = np.random.choice(range(48, len(df) - 48), size=int(frac * len(df)), replace=False)
    for i in idxs:
        if np.random.rand() > 0.5:
            df.iloc[i, df.columns.get_loc('Production')] *= np.random.uniform(1.5, 3.0)
        else:
            df.iloc[i, df.columns.get_loc('Production')] *= np.random.uniform(0.0, 0.5)
        labels[i] = 1
    df['anomaly'] = labels
    return df

def train_ffnn(X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
    inp = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def train_lstm(X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
    X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    inp = layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    x = layers.LSTM(64)(inp)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def load_data():
    """Loads and preprocesses the wind_solar.csv data."""
    try:
        df = pd.read_csv(DATA_FILE)
        if 'Date and Hour' not in df.columns:
            return None, "CRITICAL Error: 'Date and Hour' column not found in wind_solar.csv."
        df['Date and Hour'] = pd.to_datetime(df['Date and Hour'], utc=True, errors='coerce')
        if df['Date and Hour'].isna().any():
            return None, "CRITICAL Error: some 'Date and Hour' values could not be parsed as datetimes."
        df['Date and Hour'] = df['Date and Hour'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        df = df.set_index('Date and Hour').sort_index()
        if 'Production' not in df.columns:
            return None, "CRITICAL Error: 'Production' column not found in wind_solar.csv."
        if 'Source' not in df.columns:
            df['Source'] = 'Unknown'
        if 'dayName' not in df.columns:
            df['dayName'] = df.index.day_name()
        if 'monthName' not in df.columns:
            df['monthName'] = df.index.month_name()
        return df, None
    except FileNotFoundError:
        return None, "Error: wind_solar.csv not found in the data/ folder."
    except Exception as e:
        return None, f"CRITICAL Error loading or processing data: {e}"

def train_and_evaluate_models(df):
    """Train FFNN, LSTM, XGBoost and return predictions for time-series plots."""
    features = ['Production', 'hour', 'dayofweek', 'month', 'is_weekend',
                'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 'roll_mean_24', 'roll_std_24']
    N = len(df)
    train = df.iloc[:int(0.7 * N)]
    val = df.iloc[int(0.7 * N):int(0.85 * N)]
    test = df.iloc[int(0.85 * N):]
    scaler = RobustScaler().fit(train[features])
    def scale(d): return scaler.transform(d[features]), d['anomaly'].values
    X_train, y_train = scale(train)
    X_val, y_val = scale(val)
    X_test, y_test = scale(test)
    ffnn = train_ffnn(X_train, y_train, X_val, y_val)
    lstm = train_lstm(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train)
    p_ffnn = ffnn.predict(X_test).ravel()
    p_lstm = lstm.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).ravel()
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    test_results = test.copy()
    test_results['FFNN_pred'] = p_ffnn
    test_results['LSTM_pred'] = p_lstm
    test_results['XGBoost_pred'] = p_xgb
    test_results['true_anomaly'] = y_test
    return test_results, scaler

def plot_per_model_time_series_real(df, per_model_flags):
    """Plot per-model time series using real model predictions."""
    for model, preds in per_model_flags.items():
        plt.figure(figsize=(16, 8))
        plt.plot(df.index, df['Production'], label='Production', color='blue', alpha=0.7, linewidth=1)
        anomalies = df[preds == 1]
        if not anomalies.empty:
            plt.scatter(anomalies.index, anomalies['Production'],
                        label='Anomaly (Model Prediction)', color='red', s=60, marker='o', zorder=5)
        plt.title(f'Time Series with Anomalies (Real Predictions) - Model: {model}')
        plt.xlabel('Date and Hour')
        plt.ylabel('Production Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        safe_model = model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plot_path = os.path.join(PLOT_DIR, f'time_series_real_{safe_model}.png')
        plt.savefig(plot_path)
        plt.close()

def plot_confusion_matrices_real(df, per_model_flags):
    """Plot confusion matrices using real model predictions vs true anomalies."""
    y_true = df['true_anomaly']
    for model, preds in per_model_flags.items():
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        cmap = plt.cm.get_cmap('Set3')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap=cmap)
        plt.title(f'Confusion Matrix (Real) - Model: {model}')
        safe_model = model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plot_path = os.path.join(PLOT_DIR, f'confusion_matrix_real_{safe_model}.png')
        plt.savefig(plot_path)
        plt.close()

def detect_anomalies_and_plot():
    """Train models with epochs and generate per-model time-series plots with real predictions."""
    df, error = load_data()
    if error:
        return {'error': error}, None
    df = clean_and_feature_engineer(df)
    df = inject_anomalies(df, frac=0.02)
    test_results, scaler = train_and_evaluate_models(df)
    thresh = 0.5
    per_model_flags = {
        'FFNN': (test_results['FFNN_pred'] >= thresh).astype(int),
        'LSTM': (test_results['LSTM_pred'] >= thresh).astype(int),
        'Meta-Ensemble (XGBoost)': (test_results['XGBoost_pred'] >= thresh).astype(int)
    }
    plot_per_model_time_series_real(test_results, per_model_flags)
    plot_confusion_matrices_real(test_results, per_model_flags)
    return {'status': 'success'}, None

if __name__ == '__main__':
    detect_anomalies_and_plot()
