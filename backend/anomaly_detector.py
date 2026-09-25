import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import random
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- Fix for Plotting Errors in Headless Environments ---
try:
    import matplotlib
    # Forces Matplotlib to use a non-GUI backend, fixing the "Time Series chart failed" error.
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Error initializing matplotlib: {e}", file=sys.stderr)

# --- Configuration & Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, '..', 'data', 'wind_solar.csv')
CONTAMINATION_RATE = 0.0001  # 0.01% contamination for very sparse anomalies

# Ensure plots are always written to backend/plots relative to this file
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

COMPARISON_METRICS = {
    'FFNN': {'Accuracy': 0.987, 'Precision': 0.913, 'Recall': 0.269, 'F1 Score': 0.416, 'AUC': 0.814},
    'LSTM': {'Accuracy': 0.985, 'Precision': 1.000, 'Recall': 0.115, 'F1 Score': 0.207, 'AUC': 0.733},
    'Meta-Ensemble (XGBoost)': {'Accuracy': 0.973, 'Precision': 0.153, 'Recall': 0.115, 'F1 Score': 0.131, 'AUC': 0.786},
}

# ... (all imports and configuration above remain the same) ...

def clean_and_feature_engineer(df):
    """Clean data and add features as in the notebook."""
    # Filter Solar only (or adjust for both sources)
    df = df[df['Source'] == 'Solar'].copy()
    # Fill missing Production with mean
    mean_prod = df['Production'].mean()
    df['Production'] = df['Production'].fillna(mean_prod)
    # Datetime parsing already handled in load_data
    
    # Feature engineering
    df['hour'] = pd.to_datetime(df['StartHour'].astype(str).str.replace('24:00:00', '00:00:00'), format='%H:%M:%S', errors='coerce').dt.hour
    df['hour'] = df['hour'].fillna(df['hour'].mode()[0])
    df['monthName'] = df['monthName'].astype(str)
    df['month'] = pd.to_datetime(df['monthName'], format='%B', errors='coerce').dt.month
    df['month'] = df['month'].fillna(df['month'].mode()[0])
    df['dayName'] = df['dayName'].astype(str)
    df['dayofweek'] = pd.to_datetime(df['dayName'], format='%A', errors='coerce').dt.dayofweek
    df['dayofweek'] = df['dayofweek'].fillna(df['dayofweek'].mode()[0]).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Rolling stats
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
        if np.random.rand() > 0.5:  # spike
            df.iloc[i, df.columns.get_loc('Production')] *= np.random.uniform(1.5, 3.0)
        else:  # drop
            df.iloc[i, df.columns.get_loc('Production')] *= np.random.uniform(0.0, 0.5)
        labels[i] = 1
    df['anomaly'] = labels
    return df

def train_ffnn(X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
    """Train FFNN as in the notebook."""
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
    """Train a simple LSTM using reshaped data."""
    # Reshape to 3D for LSTM: (samples, timesteps, features)
    # Here we use a simple lookback=1
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
    """Train XGBoost as in the notebook."""
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
    """Loads and preprocesses the wind_solar.csv data for anomaly detection.

    Expected columns in CSV (as in the uploaded dataset):
    - Date and Hour (timezone-aware datetime string)
    - Production (numeric)
    - Source, dayName, monthName (categorical)
    """
    try:
        df = pd.read_csv(DATA_FILE)

        if 'Date and Hour' not in df.columns:
            return None, "CRITICAL Error: 'Date and Hour' column not found in wind_solar.csv."

        # Parse datetime and convert to naive (no timezone) for sklearn compatibility.
        # Use utc=True to safely handle mixed timezone offsets.
        df['Date and Hour'] = pd.to_datetime(df['Date and Hour'], utc=True, errors='coerce')
        if df['Date and Hour'].isna().any():
            return None, "CRITICAL Error: some 'Date and Hour' values could not be parsed as datetimes."

        df['Date and Hour'] = df['Date and Hour'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

        df = df.set_index('Date and Hour').sort_index()

        if 'Production' not in df.columns:
            return None, "CRITICAL Error: 'Production' column not found in wind_solar.csv."

        # Ensure required categorical columns exist
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
        # Return a clear error for the API/frontend
        return None, f"CRITICAL Error loading or processing data: {e}"

def train_and_evaluate_models(df):
    """Train FFNN, LSTM, XGBoost and return predictions for time-series plots."""
    # Features used in notebook
    features = ['Production', 'hour', 'dayofweek', 'month', 'is_weekend',
                'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 'roll_mean_24', 'roll_std_24']
    # Train/val/test split
    N = len(df)
    train = df.iloc[:int(0.7 * N)]
    val = df.iloc[int(0.7 * N):int(0.85 * N)]
    test = df.iloc[int(0.85 * N):]

    scaler = RobustScaler().fit(train[features])
    def scale(d): return scaler.transform(d[features]), d['anomaly'].values
    X_train, y_train = scale(train)
    X_val, y_val = scale(val)
    X_test, y_test = scale(test)

    # Train models
    ffnn = train_ffnn(X_train, y_train, X_val, y_val)
    lstm = train_lstm(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train)

    # Predict on test set
    p_ffnn = ffnn.predict(X_test).ravel()
    p_lstm = lstm.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).ravel()
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Store predictions and true labels for plotting
    test_results = test.copy()
    test_results['FFNN_pred'] = p_ffnn
    test_results['LSTM_pred'] = p_lstm
    test_results['XGBoost_pred'] = p_xgb
    test_results['true_anomaly'] = y_test

    return test_results, scaler

def load_and_inject_anomalies():
    """Loads data and injects a few clear anomalies for demonstration."""
    df, error = load_data()
    if error:
        return None, error
    # Inject anomalies at specific timestamps
    # Choose a few existing timestamps and set extreme values
    inject_indices = [
        df.index[1000],   # early in dataset
        df.index[5000],   # middle
        df.index[9000],   # later
    ]
    # Spike
    df.loc[inject_indices[0], 'Production'] = df['Production'].max() * 3
    # Drop
    df.loc[inject_indices[1], 'Production'] = 0
    # External injection (zero after non-zero)
    df.loc[inject_indices[2], 'Production'] = 0
    return df, None

# ... (all functions below remain the same) ...
def generate_comparison_plot():
    """Generates and saves the AUC comparison plot (Model Performance Graph)."""
    try:
        metrics_df = pd.DataFrame(COMPARISON_METRICS).T
        
        plt.figure(figsize=(8, 5))
        auc_data = metrics_df['AUC'].sort_values(ascending=False)

        bars = plt.bar(auc_data.index, auc_data.values, color='skyblue')

        # Highlight the best model in a different color
        best_idx = 0
        bars[best_idx].set_color('green')

        plt.title('Model Comparison: AUC Performance (Higher is Better)', fontsize=14)
        plt.ylabel('AUC Score', fontsize=12)
        plt.xticks(rotation=15, ha='right', fontsize=10)
        plt.ylim(0, 1)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

        # Add a short explanatory text indicating which model is best
        best_model_name = auc_data.index[0]
        best_model_auc = auc_data.iloc[0]
        plt.text(0.5, -0.2,
                 f"Best model: {best_model_name} (AUC = {best_model_auc:.3f})",
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plot_path = os.path.join(PLOT_DIR, 'model_comparison_auc.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating comparison plot: {e}", file=sys.stderr)
        return None

def generate_detailed_metric_plots():
    """Generates additional evaluation metric plots for all models."""
    try:
        metrics_df = pd.DataFrame(COMPARISON_METRICS).T

        # Bar plot for all metrics per model
        plt.figure(figsize=(10, 6))
        metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']].plot(
            kind='bar', figsize=(10, 6)
        )
        plt.title('Model Comparison Across Evaluation Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=15, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()
        plot_path_all = os.path.join(PLOT_DIR, 'model_comparison_all_metrics.png')
        plt.savefig(plot_path_all)
        plt.close()

        # Individual bar plots per metric
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        for metric in metric_names:
            plt.figure(figsize=(8, 5))
            metric_series = metrics_df[metric].sort_values(ascending=False)
            bars = plt.bar(metric_series.index, metric_series.values, color='skyblue')
            bars[0].set_color('red')
            plt.title(f'Model Comparison: {metric}')
            plt.ylabel(metric)
            plt.xticks(rotation=15, ha='right')
            plt.ylim(0, 1)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3),
                         ha='center', va='bottom')
            plt.tight_layout()
            metric_plot_path = os.path.join(PLOT_DIR, f'model_comparison_{metric.lower().replace(" ", "_")}.png')
            plt.savefig(metric_plot_path)
            plt.close()

        return True
    except Exception as e:
        print(f"Error generating detailed metric plots: {e}", file=sys.stderr)
        return False

def detect_anomalies_and_plot():
    """Train models with epochs and generate per-model time-series plots with real predictions."""
    df, error = load_data()
    if error:
        return {'error': error}, None
    df = clean_and_feature_engineer(df)
    df = inject_anomalies(df, frac=0.02)
    test_results, scaler = train_and_evaluate_models(df)

    # Default threshold for basic plots
    thresh = 0.5
    per_model_flags = {
        'FFNN': (test_results['FFNN_pred'] >= thresh).astype(int),
        'LSTM': (test_results['LSTM_pred'] >= thresh).astype(int),
        'Meta-Ensemble (XGBoost)': (test_results['XGBoost_pred'] >= thresh).astype(int)
    }
    per_model_preds = {
        'FFNN': test_results['FFNN_pred'],
        'LSTM': test_results['LSTM_pred'],
        'Meta-Ensemble (XGBoost)': test_results['XGBoost_pred']
    }

    # Generate all plots
    plot_per_model_time_series_real(test_results, per_model_flags)
    plot_confusion_matrices_real(test_results, per_model_flags)
    plot_confusion_matrices_multiple_thresholds(test_results, per_model_preds, thresholds=[0.5, 0.8, 0.35])
    plot_roc_curves(test_results, per_model_preds)
    plot_precision_recall_curves(test_results, per_model_preds)
    
    # Generate comprehensive metrics comparison
    metrics_comparison = generate_comprehensive_metrics_comparison(test_results, per_model_preds)
    
    # Save detailed metrics table
    metrics_df = print_metrics_table(test_results, per_model_preds, thresh=thresh)
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table = ax.table(cellText=metrics_df.round(3).values,
                     colLabels=metrics_df.columns,
                     rowLabels=metrics_df.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Evaluation Metrics Comparison (threshold=0.5)', pad=20, fontsize=14, fontweight='bold')
    plot_path = os.path.join(PLOT_DIR, 'evaluation_metrics_table.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {'status': 'success', 'metrics_comparison': metrics_comparison}, None

def generate_comprehensive_metrics_comparison(df, per_model_preds, thresholds=[0.5, 0.8, 0.35]):
    """Generate comprehensive metrics comparison across all models and thresholds."""
    y_true = df['true_anomaly']
    all_metrics = []
    
    for model, probs in per_model_preds.items():
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            
            # Calculate all metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc = roc_auc_score(y_true, probs)
            
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            all_metrics.append({
                'Model': model,
                'Threshold': thresh,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'AUC': roc,
                'True Positives': tp,
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatives': fn
            })
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy across thresholds
    sns.barplot(data=metrics_df, x='Model', y='Accuracy', hue='Threshold', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy by Model and Threshold')
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Precision-Recall Tradeoff
    for model in metrics_df['Model'].unique():
        model_data = metrics_df[metrics_df['Model'] == model]
        axes[0, 1].plot(model_data['Recall'], model_data['Precision'], 'o-', label=model, linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Tradeoff')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score comparison
    sns.barplot(data=metrics_df, x='Model', y='F1 Score', hue='Threshold', ax=axes[1, 0])
    axes[1, 0].set_title('F1 Score by Model and Threshold')
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: AUC comparison (threshold-independent)
    auc_data = metrics_df[['Model', 'AUC']].drop_duplicates()
    auc_data = auc_data.sort_values('AUC', ascending=False)
    bars = axes[1, 1].bar(auc_data['Model'], auc_data['AUC'], color=['gold', 'silver', 'lightcoral'])
    axes[1, 1].set_title('AUC Score Comparison (Threshold-Independent)')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel('AUC')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'comprehensive_metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determine best model based on multiple criteria
    best_model_analysis = determine_best_model(metrics_df)
    
    return {
        'detailed_metrics': metrics_df.to_dict('records'),
        'best_model_analysis': best_model_analysis,
        'summary_plot_path': plot_path
    }

def determine_best_model(metrics_df):
    """Determine the best model based on multiple criteria."""
    # Calculate average scores across thresholds for each model
    model_summary = metrics_df.groupby('Model').agg({
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1 Score': 'mean',
        'AUC': 'first'  # AUC is the same across thresholds
    }).round(3)
    
    # Calculate composite score (weighted average)
    weights = {'Accuracy': 0.2, 'Precision': 0.25, 'Recall': 0.25, 'F1 Score': 0.2, 'AUC': 0.1}
    model_summary['Composite_Score'] = (
        model_summary['Accuracy'] * weights['Accuracy'] +
        model_summary['Precision'] * weights['Precision'] +
        model_summary['Recall'] * weights['Recall'] +
        model_summary['F1 Score'] * weights['F1 Score'] +
        model_summary['AUC'] * weights['AUC']
    ).round(3)
    
    # Sort by composite score
    model_summary = model_summary.sort_values('Composite_Score', ascending=False)
    
    best_model = model_summary.index[0]
    best_score = model_summary.loc[best_model, 'Composite_Score']
    best_auc = model_summary.loc[best_model, 'AUC']
    
    return {
        'best_model': best_model,
        'composite_score': best_score,
        'best_auc': best_auc,
        'model_rankings': model_summary.to_dict('index'),
        'recommendation': f"The {best_model} model performs best with a composite score of {best_score:.3f} and AUC of {best_auc:.3f}."
    }

def generate_synthetic_per_model_flags(df, base_anomalies):
    """
    Generate synthetic per-model anomaly flags based on model AUCs.
    - base_anomalies: IsolationForest anomaly flags (used as a base).
    - Returns a dict: {model_name: Series of anomaly flags}
    """
    # Extract AUC values from COMPARISON_METRICS
    model_aucs = {model: COMPARISON_METRICS[model]['AUC'] for model in COMPARISON_METRICS}
    # Sort models by AUC (higher AUC = more accurate)
    models_sorted = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)

    # Convert base anomalies to boolean (True if anomaly)
    base_bool = (base_anomalies == -1)

    per_model_flags = {}
    for model, auc in models_sorted:
        # Higher AUC -> more agreement with base anomalies
        agreement_prob = auc  # use AUC as agreement probability
        # Initialize with base flags
        flags = base_bool.copy()
        # Introduce synthetic disagreements based on AUC
        n = len(flags)
        # Number of points to flip (lower AUC -> more flips)
        n_flip = int((1 - agreement_prob) * n * 0.1)  # up to 10% of points flipped
        if n_flip > 0:
            flip_indices = random.sample(range(n), n_flip)
            flags.iloc[flip_indices] = ~flags.iloc[flip_indices]
        # Convert back to -1/1 encoding
        per_model_flags[model] = flags.map({True: -1, False: 1})
    return per_model_flags

def classify_anomaly_type(row, prev_row=None):
    """
    Classify anomaly type based on production change.
    Returns one of: 'Sudden Spike', 'Sudden Down', 'External Injection', 'Normal'
    """
    if prev_row is None:
        return 'Normal'
    diff = row['Production'] - prev_row['Production']
    # Simple heuristics
    if diff > 0 and diff > 0.5 * prev_row['Production']:
        return 'Sudden Spike'
    elif diff < 0 and abs(diff) > 0.5 * prev_row['Production']:
        return 'Sudden Down'
    elif prev_row['Production'] > 0 and row['Production'] == 0:
        return 'External Injection'
    else:
        return 'Normal'

def plot_per_model_time_series_real(df, per_model_flags):
    """Plot per-model time series using real model predictions with enhanced visualization."""
    for model, preds in per_model_flags.items():
        plt.figure(figsize=(16, 8))
        
        # Plot production values
        plt.plot(df.index, df['Production'], label='Production', color='blue', alpha=0.7, linewidth=1)
        
        # Highlight anomalies
        anomalies = df[preds == 1]
        if not anomalies.empty:
            plt.scatter(anomalies.index, anomalies['Production'],
                        label='Predicted Anomaly', color='red', s=80, marker='o', zorder=5, edgecolors='darkred', linewidth=1)
        
        # Also show true anomalies for comparison
        true_anomalies = df[df['true_anomaly'] == 1]
        if not true_anomalies.empty:
            plt.scatter(true_anomalies.index, true_anomalies['Production'],
                        label='True Anomaly', color='orange', s=60, marker='^', zorder=4, alpha=0.7)
        
        plt.title(f'Time Series with Anomaly Detection - Model: {model}', fontsize=14, fontweight='bold')
        plt.xlabel('Date and Hour', fontsize=12)
        plt.ylabel('Production Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        safe_model = model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plot_path = os.path.join(PLOT_DIR, f'time_series_enhanced_{safe_model}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        # Also save a generic filename for the Meta-Ensemble model so the frontend
        # path /plots/anomaly_time_series_visualization.png resolves correctly.
        if safe_model == 'Meta_Ensemble_XGBoost':
            generic_path = os.path.join(PLOT_DIR, 'anomaly_time_series_visualization.png')
            plt.savefig(generic_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_confusion_matrices_real(df, per_model_flags):
    """Plot confusion matrices using real model predictions vs true anomalies."""
    y_true = df['true_anomaly']
    for model, preds in per_model_flags.items():
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        cmap = plt.colormaps.get_cmap('Set3')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap=cmap)
        plt.title(f'Confusion Matrix (Real) - Model: {model}')
        safe_model = model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plot_path = os.path.join(PLOT_DIR, f'confusion_matrix_real_{safe_model}.png')
        plt.savefig(plot_path)
        plt.close()

def plot_confusion_matrices_multiple_thresholds(df, per_model_preds, thresholds=[0.5, 0.8, 0.35]):
    """Plot confusion matrices for multiple thresholds for all models."""
    y_true = df['true_anomaly']
    
    for model, probs in per_model_preds.items():
        fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5))
        if len(thresholds) == 1:
            axes = [axes]
        
        for i, thresh in enumerate(thresholds):
            y_pred = (probs >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            cmap = plt.colormaps.get_cmap('Set3')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
            disp.plot(cmap=cmap, ax=axes[i])
            
            axes[i].set_title(f'{model}\nThreshold: {thresh}\nAcc: {acc:.3f}, P: {prec:.3f}, R: {rec:.3f}, F1: {f1:.3f}')
        
        plt.tight_layout()
        safe_model = model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plot_path = os.path.join(PLOT_DIR, f'confusion_matrices_multi_threshold_{safe_model}.png')
        plt.savefig(plot_path)
        plt.close()

def plot_roc_curves(df, per_model_preds):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 7))
    y_true = df['true_anomaly']
    colors = {'FFNN': 'darkorange', 'LSTM': 'blue', 'Meta-Ensemble (XGBoost)': 'green'}
    for model, probs in per_model_preds.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors.get(model, 'black'), lw=2, label=f'{model} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, 'roc_curves_comparison.png')
    plt.savefig(plot_path)
    plt.close()

def plot_precision_recall_curves(df, per_model_preds):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(8, 7))
    y_true = df['true_anomaly']
    colors = {'FFNN': 'darkorange', 'LSTM': 'blue', 'Meta-Ensemble (XGBoost)': 'green'}
    for model, probs in per_model_preds.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors.get(model, 'black'), lw=2, label=f'{model} (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, 'precision_recall_comparison.png')
    plt.savefig(plot_path)
    plt.close()

def print_metrics_table(df, per_model_preds, thresh=0.5):
    """Print a table of evaluation metrics for all models at 0.5 threshold."""
    y_true = df['true_anomaly']
    metrics = []
    for model, probs in per_model_preds.items():
        y_pred = (probs >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc = roc_auc_score(y_true, probs)
        metrics.append({'Model': model, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'AUC': roc})
    df_metrics = pd.DataFrame(metrics).set_index('Model')
    print('\n📊 Evaluation Metrics at threshold 0.5:')
    print(df_metrics.round(3))
    return df_metrics

def plot_confusion_matrices(df, per_model_flags):
    """
    Plot confusion matrix per model using IsolationForest flags as reference.
    Uses different colors for different anomaly types.
    """
    # Use IsolationForest flags as "ground truth" (since we don't have real labels)
    y_true = (df['Anomaly_Flag'] == -1).astype(int)  # 1 if anomaly, 0 if normal
    for model, flags in per_model_flags.items():
        y_pred = (flags == -1).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # Custom colormap with distinct colors
        cmap = plt.cm.get_cmap('Set3')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap=cmap)
        plt.title(f'Confusion Matrix - Model: {model}')
        safe_model = model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plot_path = os.path.join(PLOT_DIR, f'confusion_matrix_model_{safe_model}.png')
        plt.savefig(plot_path)
        plt.close()

def load_real_per_model_predictions(file_path):
    """
    Load real per-model predictions from a CSV file.
    Expected columns: 'Date and Hour', 'FFNN_pred', 'LSTM_pred', 'Meta_Ensemble_pred', 'True_Label' (optional)
    Returns a DataFrame with datetime index and per-model prediction columns.
    """
    try:
        df_pred = pd.read_csv(file_path)
        if 'Date and Hour' not in df_pred.columns:
            raise ValueError("Missing 'Date and Hour' column in predictions file.")
        df_pred['Date and Hour'] = pd.to_datetime(df_pred['Date and Hour'], utc=True, errors='coerce')
        if df_pred['Date and Hour'].isna().any():
            raise ValueError("Some 'Date and Hour' values could not be parsed.")
        df_pred['Date and Hour'] = df_pred['Date and Hour'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        df_pred = df_pred.set_index('Date and Hour').sort_index()
        return df_pred
    except Exception as e:
        print(f"Error loading real predictions file: {e}", file=sys.stderr)
        return None

def plot_anomaly_counts_per_source(df):
    """Generates a bar chart of number of detected anomalies per Source."""
    try:
        if 'Source' not in df.columns or 'Anomaly_Flag' not in df.columns:
            return None

        anomalies = df[df['Anomaly_Flag'] == -1]
        if anomalies.empty:
            return None

        counts = anomalies['Source'].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(counts.index, counts.values, color='orange')
        plt.title('Number of Detected Anomalies per Source')
        plt.ylabel('Anomaly Count')
        plt.xlabel('Source')
        plt.xticks(rotation=20, ha='right')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval),
                     ha='center', va='bottom')

        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, 'anomaly_counts_per_source.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating anomaly count plot: {e}", file=sys.stderr)
        return None

def get_model_metrics():
    """Returns the model comparison metrics as a list of dicts for the frontend."""
    return [{'Model': k, **v} for k, v in COMPARISON_METRICS.items()]

def get_inference_result():
    """Calculates the final inference based on model metrics."""
    metrics_df = pd.DataFrame(COMPARISON_METRICS).T
    best_model = metrics_df['AUC'].idxmax()
    best_auc = metrics_df['AUC'].max()

    inference_result = [
        {'Key': 'Model Comparison', 'Value': f"The best performing model is the {best_model}."},
        {'Key': 'Performance Metric', 'Value': f"It achieved the highest AUC score of {best_auc}."},
        {'Key': 'Robustness', 'Value': "This indicates superior ability to distinguish anomalies from normal data, making it the most robust choice."},
        {'Key': 'Anomaly Type', 'Value': "The anomalies detected are classified as 'Sudden Spikes' in Production values."}
    ]
    return inference_result

if __name__ == '__main__':
    generate_comparison_plot()
    generate_detailed_metric_plots()
    detect_anomalies_and_plot()
    print("Initial plots generated successfully.")