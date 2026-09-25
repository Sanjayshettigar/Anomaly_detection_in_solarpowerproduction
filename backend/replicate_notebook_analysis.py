#!/usr/bin/env python3
"""Replicate the exact notebook methodology for anomaly detection."""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set(style="whitegrid")
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import xgboost as xgb
import os
import sys

# Set up plots directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def inject_anomalies(data, frac=0.02, seed=42):
    """Exact replica of notebook anomaly injection."""
    np.random.seed(seed)
    df = data.copy()
    labels = np.zeros(len(df))
    
    # Use iloc for indexing to avoid issues with datetime index
    idxs = np.random.choice(range(48, len(df)-48), size=int(frac*len(df)), replace=False)
    
    for i in idxs:
        if np.random.rand()>0.5:   # spike
            df.iloc[i, df.columns.get_loc('Production')] *= np.random.uniform(1.5,3.0)
        else:                      # drop
            df.iloc[i, df.columns.get_loc('Production')] *= np.random.uniform(0.0,0.5)
        labels[i]=1
    
    df['anomaly']=labels
    return df

def train_ffnn_notebook(X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
    """Exact FFNN architecture from notebook."""
    inp=layers.Input(shape=(X_train.shape[1],))
    x=layers.Dense(128,activation='relu')(inp)
    x=layers.Dropout(0.2)(x)
    x=layers.Dense(64,activation='relu')(x)
    out=layers.Dense(1,activation='sigmoid')(x)
    ffnn=models.Model(inp,out)
    ffnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
    ffnn.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=batch_size,verbose=1)
    return ffnn

def train_lstm_notebook(X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
    """Train LSTM with same architecture as notebook."""
    # Reshape for LSTM
    X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    
    inp=layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    x=layers.LSTM(64)(inp)
    x=layers.Dropout(0.2)(x)
    out=layers.Dense(1,activation='sigmoid')(x)
    lstm=models.Model(inp,out)
    lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
    lstm.fit(X_train_seq,y_train,validation_data=(X_val_seq,y_val),epochs=epochs,batch_size=batch_size,verbose=1)
    return lstm

def train_xgboost_notebook(X_train, y_train):
    """Train XGBoost with same parameters as notebook."""
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

def preprocess_data_notebook_style(df):
    """Preprocess data exactly as in notebook."""
    # Filter for Solar data only (as in notebook)
    df = df[df['Source'] == 'Solar'].copy()
    
    # Fill missing Production with mean (as in notebook)
    mean_prod = df['Production'].mean()
    df['Production'] = df['Production'].fillna(mean_prod)
    
    # Feature engineering exactly as notebook
    df['hour'] = pd.to_datetime(df['StartHour'], format='%H:%M:%S', errors='coerce').dt.hour
    df['hour'] = df['hour'].fillna(df['hour'].mode()[0])
    
    df['monthName'] = df['monthName'].astype(str)
    df['month'] = pd.to_datetime(df['monthName'], format='%B', errors='coerce').dt.month
    df['month'] = df['month'].fillna(df['month'].mode()[0])
    
    df['dayName'] = df['dayName'].astype(str)
    df['dayofweek'] = pd.to_datetime(df['dayName'], format='%A', errors='coerce').dt.dayofweek
    df['dayofweek'] = df['dayofweek'].fillna(df['dayofweek'].mode()[0]).astype(int)
    
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    
    # Rolling statistics exactly as notebook
    for w in [3,6,24]:
        df[f'roll_mean_{w}'] = df['Production'].rolling(w).mean().shift(1)
        df[f'roll_std_{w}'] = df['Production'].rolling(w).std().shift(1)
    
    # Fill missing values (notebook uses deprecated method, we'll use modern equivalent)
    df = df.ffill().fillna(0)
    
    return df

def plot_time_series_with_anomalies(df, predictions, model_name, threshold=0.5):
    """Create time series plot like in notebook."""
    plt.figure(figsize=(16, 8))
    
    # Plot production
    plt.plot(df.index, df['Production'], label='Production', color='blue', alpha=0.7, linewidth=1)
    
    # Plot predicted anomalies
    pred_anomalies = (predictions >= threshold).astype(int)
    anomaly_points = df[pred_anomalies == 1]
    
    if not anomaly_points.empty:
        plt.scatter(anomaly_points.index, anomaly_points['Production'],
                   label=f'{model_name} Detected Anomalies (Threshold > {threshold})',
                   color='red', s=60, marker='o', zorder=5)
    
    plt.title(f'Detected Anomalies in Production Data ({model_name}) - Threshold Adjusted to {threshold}')
    plt.xlabel('Date and Hour')
    plt.ylabel('Production')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    plot_path = os.path.join(PLOT_DIR, f'time_series_{safe_name}_threshold_{threshold}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_multi_threshold(y_true, y_pred_probs, model_name, thresholds=[0.5, 0.8, 0.35]):
    """Plot confusion matrices for multiple thresholds."""
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5))
    if len(thresholds) == 1:
        axes = [axes]
    
    for i, thresh in enumerate(thresholds):
        y_pred = (y_pred_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(cmap=plt.cm.Set3, ax=axes[i])
        
        axes[i].set_title(f'{model_name}\nThreshold: {thresh}\nAcc: {acc:.3f}, P: {prec:.3f}, R: {rec:.3f}, F1: {f1:.3f}')
    
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    plot_path = os.path.join(PLOT_DIR, f'confusion_matrices_{safe_name}_multi_threshold.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all_models():
    """Main evaluation function following notebook methodology."""
    print("🚀 Starting Notebook-Style Anomaly Detection Analysis...")
    
    # Load and preprocess data
    data_path = os.path.join(BASE_DIR, '..', 'data', 'wind_solar.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('C:', 'Users', 'SACHIN N S', 'Downloads', 'data', 'wind_solar.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('C:', 'Users', 'SACHIN N S', 'Downloads', 'wind_solar.csv')
    df = pd.read_csv(data_path, parse_dates=['Date and Hour'])
    
    # Preprocess exactly like notebook
    df_processed = preprocess_data_notebook_style(df)
    
    # Inject anomalies exactly like notebook
    df_anomalies = inject_anomalies(df_processed, frac=0.02, seed=42)
    print(f"Injected anomalies: {df_anomalies['anomaly'].sum()}")
    
    # Train/val/test split exactly like notebook (70%/15%/15%)
    N = len(df_anomalies)
    train = df_anomalies.iloc[:int(0.7 * N)]
    val = df_anomalies.iloc[int(0.7 * N):int(0.85 * N)]
    test = df_anomalies.iloc[int(0.85 * N):]
    
    # Features exactly like notebook
    features = ['Production', 'hour', 'dayofweek', 'month', 'is_weekend',
                'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 
                'roll_mean_24', 'roll_std_24']
    
    # Scale data exactly like notebook
    scaler = RobustScaler().fit(train[features])
    def scale(d): return scaler.transform(d[features]), d['anomaly'].values
    
    X_train, y_train = scale(train)
    X_val, y_val = scale(val)
    X_test, y_test = scale(test)
    
    print("📊 Training FFNN...")
    ffnn = train_ffnn_notebook(X_train, y_train, X_val, y_val)
    p_ffnn = ffnn.predict(X_test).ravel()
    
    print("📊 Training LSTM...")
    lstm = train_lstm_notebook(X_train, y_train, X_val, y_val)
    X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    p_lstm = lstm.predict(X_test_seq).ravel()
    
    print("📊 Training XGBoost...")
    xgb_model = train_xgboost_notebook(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Store results with datetime index
    test_results = test.copy()
    test_results['FFNN_pred'] = p_ffnn
    test_results['LSTM_pred'] = p_lstm
    test_results['XGBoost_pred'] = p_xgb
    test_results['true_anomaly'] = y_test
    
    # Generate time series plots for each model
    print("📈 Generating time series plots...")
    plot_time_series_with_anomalies(test_results, p_ffnn, 'FFNN', threshold=0.5)
    plot_time_series_with_anomalies(test_results, p_lstm, 'LSTM', threshold=0.5)
    plot_time_series_with_anomalies(test_results, p_xgb, 'Meta-Ensemble XGBoost', threshold=0.5)
    
    # Generate multi-threshold confusion matrices
    print("📊 Generating confusion matrices for multiple thresholds...")
    plot_confusion_matrix_multi_threshold(y_test, p_ffnn, 'FFNN', thresholds=[0.5, 0.8, 0.35])
    plot_confusion_matrix_multi_threshold(y_test, p_lstm, 'LSTM', thresholds=[0.5, 0.8, 0.35])
    plot_confusion_matrix_multi_threshold(y_test, p_xgb, 'Meta-Ensemble XGBoost', thresholds=[0.5, 0.8, 0.35])
    
    # Calculate and display comprehensive metrics
    print("\n🏆 COMPREHENSIVE MODEL EVALUATION:")
    print("=" * 60)
    
    models_data = {
        'FFNN': p_ffnn,
        'LSTM': p_lstm,
        'Meta-Ensemble XGBoost': p_xgb
    }
    
    results = []
    for model_name, predictions in models_data.items():
        print(f"\n📊 {model_name} Results:")
        print("-" * 40)
        
        for thresh in [0.5, 0.8, 0.35]:
            y_pred = (predictions >= thresh).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, predictions)
            
            print(f"  Threshold {thresh}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
            
            results.append({
                'Model': model_name,
                'Threshold': thresh,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'AUC': auc
            })
    
    # Determine best model
    results_df = pd.DataFrame(results)
    
    # Calculate average scores across thresholds
    model_summary = results_df.groupby('Model').agg({
        'Accuracy': 'mean',
        'Precision': 'mean', 
        'Recall': 'mean',
        'F1 Score': 'mean',
        'AUC': 'first'  # AUC is same across thresholds
    }).round(3)
    
    # Calculate composite score (same weights as notebook methodology)
    weights = {'Accuracy': 0.2, 'Precision': 0.25, 'Recall': 0.25, 'F1 Score': 0.2, 'AUC': 0.1}
    model_summary['Composite_Score'] = (
        model_summary['Accuracy'] * weights['Accuracy'] +
        model_summary['Precision'] * weights['Precision'] +
        model_summary['Recall'] * weights['Recall'] +
        model_summary['F1 Score'] * weights['F1 Score'] +
        model_summary['AUC'] * weights['AUC']
    ).round(3)
    
    model_summary = model_summary.sort_values('Composite_Score', ascending=False)
    
    print("\n🏆 FINAL MODEL RANKINGS:")
    print("=" * 50)
    print(model_summary)
    
    best_model = model_summary.index[0]
    best_score = model_summary.loc[best_model, 'Composite_Score']
    best_auc = model_summary.loc[best_model, 'AUC']
    
    print(f"\n🥇 BEST MODEL: {best_model}")
    print(f"📈 Composite Score: {best_score}")
    print(f"🎯 AUC Score: {best_auc}")
    print(f"💡 RECOMMENDATION: The {best_model} model performs best with composite score {best_score:.3f} and AUC {best_auc:.3f}")
    
    return {
        'best_model': best_model,
        'composite_score': best_score,
        'auc_score': best_auc,
        'detailed_results': results_df.to_dict('records'),
        'model_summary': model_summary.to_dict('index')
    }

if __name__ == "__main__":
    results = evaluate_all_models()
    print(f"\n🎉 Analysis completed! Best model: {results['best_model']}")
