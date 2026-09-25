#!/usr/bin/env python3
"""Comprehensive Model Comparison focusing on AUC and all evaluation metrics."""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set(style="whitegrid")
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
import os

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
    
    # Fill missing values
    df = df.ffill().fillna(0)
    
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
    ffnn.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=batch_size,verbose=0)
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
    lstm.fit(X_train_seq,y_train,validation_data=(X_val_seq,y_val),epochs=epochs,batch_size=batch_size,verbose=0)
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

def plot_roc_curves_comparison(y_true, model_predictions, model_names):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (preds, name) in enumerate(zip(model_predictions, model_names)):
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_score = roc_auc_score(y_true, preds)
        
        plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, 'roc_curves_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves_comparison(y_true, model_predictions, model_names):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (preds, name) in enumerate(zip(model_predictions, model_names)):
        precision, recall, _ = precision_recall_curve(y_true, preds)
        avg_precision = average_precision_score(y_true, preds)
        
        plt.plot(recall, precision, color=colors[i], linewidth=2,
                label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, 'precision_recall_curves_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_auc_comparison(model_names, auc_scores):
    """Plot AUC comparison bar chart."""
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(model_names, auc_scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('AUC Score Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, 'auc_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_metrics_comparison(metrics_df):
    """Plot comprehensive comparison of all metrics across models."""
    # Melt the dataframe for easier plotting
    metrics_melted = metrics_df.melt(id_vars=['Model', 'Threshold'], 
                                     var_name='Metric', 
                                     value_name='Score')
    
    # Create separate plots for each threshold
    thresholds = metrics_df['Threshold'].unique()
    
    for thresh in thresholds:
        plt.figure(figsize=(16, 10))
        
        thresh_data = metrics_melted[metrics_melted['Threshold'] == thresh]
        
        # Create subplots for different metric groups
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Metrics Comparison - Threshold {thresh}', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        for i, metric in enumerate(metrics_to_plot):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            metric_data = thresh_data[thresh_data['Metric'] == metric]
            
            # Create bar plot
            models = metric_data['Model'].unique()
            scores = metric_data['Score'].values
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)]
            bars = ax.bar(models, scores, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(models) > 2:
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f'comprehensive_metrics_comparison_threshold_{thresh}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_f1_score_trends(metrics_df):
    """Plot F1 score trends across thresholds."""
    plt.figure(figsize=(12, 8))
    
    models = metrics_df['Model'].unique()
    thresholds = sorted(metrics_df['Threshold'].unique())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for i, model in enumerate(models):
        model_data = metrics_df[metrics_df['Model'] == model]
        f1_scores = model_data.set_index('Threshold')['F1 Score'].reindex(thresholds)
        
        plt.plot(thresholds, f1_scores, color=colors[i], marker=markers[i], 
                linewidth=2, markersize=8, label=model)
        
        # Add value labels
        for j, (thresh, f1) in enumerate(f1_scores.items()):
            plt.annotate(f'{f1:.3f}', (thresh, f1), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Trends Across Thresholds', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, 'f1_score_trends.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def comprehensive_auc_analysis():
    """Main function for comprehensive AUC-focused analysis."""
    print("🚀 Starting Comprehensive AUC and Metrics Analysis...")
    
    # Load and preprocess data
    data_path = os.path.join(BASE_DIR, '..', 'data', 'wind_solar.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('C:', 'Users', 'SACHIN N S', 'Downloads', 'wind_solar.csv')
    
    df = pd.read_csv(data_path, parse_dates=['Date and Hour'])
    df_processed = preprocess_data_notebook_style(df)
    df_anomalies = inject_anomalies(df_processed, frac=0.02, seed=42)
    
    print(f"📊 Dataset prepared with {df_anomalies['anomaly'].sum()} injected anomalies")
    
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
    
    print("📊 Training models...")
    
    # Train models
    ffnn = train_ffnn_notebook(X_train, y_train, X_val, y_val)
    p_ffnn = ffnn.predict(X_test).ravel()
    
    lstm = train_lstm_notebook(X_train, y_train, X_val, y_val)
    X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    p_lstm = lstm.predict(X_test_seq).ravel()
    
    xgb_model = train_xgboost_notebook(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Prepare data for analysis
    model_names = ['FFNN', 'LSTM', 'Meta-Ensemble XGBoost']
    model_predictions = [p_ffnn, p_lstm, p_xgb]
    
    # Calculate AUC scores
    auc_scores = [roc_auc_score(y_test, preds) for preds in model_predictions]
    
    print("\n🎯 AUC SCORES COMPARISON:")
    print("=" * 40)
    for name, auc in zip(model_names, auc_scores):
        print(f"{name}: {auc:.3f}")
    
    # Generate ROC curves
    print("📈 Generating ROC curves...")
    plot_roc_curves_comparison(y_test, model_predictions, model_names)
    
    # Generate Precision-Recall curves
    print("📈 Generating Precision-Recall curves...")
    plot_precision_recall_curves_comparison(y_test, model_predictions, model_names)
    
    # Generate AUC comparison bar chart
    print("📊 Generating AUC comparison chart...")
    plot_auc_comparison(model_names, auc_scores)
    
    # Calculate comprehensive metrics for all thresholds
    thresholds = [0.5, 0.8, 0.35]
    all_metrics = []
    
    for model_name, predictions in zip(model_names, model_predictions):
        for thresh in thresholds:
            y_pred = (predictions >= thresh).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, predictions)
            
            all_metrics.append({
                'Model': model_name,
                'Threshold': thresh,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'AUC': auc
            })
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Generate comprehensive metrics comparison
    print("📊 Generating comprehensive metrics comparison...")
    plot_comprehensive_metrics_comparison(metrics_df)
    
    # Generate F1 score trends
    print("📈 Generating F1 score trends...")
    plot_f1_score_trends(metrics_df)
    
    # Determine best model based on AUC
    best_auc_idx = np.argmax(auc_scores)
    best_model_auc = model_names[best_auc_idx]
    best_auc_score = auc_scores[best_auc_idx]
    
    print(f"\n🏆 BEST MODEL BY AUC: {best_model_auc}")
    print(f"📈 AUC Score: {best_auc_score:.3f}")
    
    # Create summary table
    print("\n📋 COMPREHENSIVE METRICS TABLE:")
    print("=" * 80)
    print(metrics_df.round(3).to_string(index=False))
    
    return {
        'best_model_auc': best_model_auc,
        'best_auc_score': best_auc_score,
        'all_metrics': metrics_df.to_dict('records'),
        'auc_scores': dict(zip(model_names, auc_scores))
    }

if __name__ == "__main__":
    results = comprehensive_auc_analysis()
    print(f"\n🎉 Analysis completed! Best model by AUC: {results['best_model_auc']} with score {results['best_auc_score']:.3f}")
