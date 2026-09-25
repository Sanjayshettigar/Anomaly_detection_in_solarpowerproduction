from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# Create plots directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

class AnomalyDetectionAPI:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.test_data = None
        self.results = {}
        
    def load_and_train_models(self):
        """Load data and train all models"""
        try:
            # Load data
            data_path = os.path.join(DATA_DIR, 'wind_solar.csv')
            if not os.path.exists(data_path):
                data_path = os.path.join('C:', 'Users', 'SACHIN N S', 'Downloads', 'wind_solar.csv')
            
            df = pd.read_csv(data_path, parse_dates=['Date and Hour'])
            
            # Preprocess
            df = self.preprocess_data(df)
            df_anomalies = self.inject_anomalies(df)
            
            # Train/test split
            N = len(df_anomalies)
            train = df_anomalies.iloc[:int(0.7 * N)]
            val = df_anomalies.iloc[int(0.7 * N):int(0.85 * N)]
            self.test_data = df_anomalies.iloc[int(0.85 * N):]
            
            # Features
            features = ['Production', 'hour', 'dayofweek', 'month', 'is_weekend',
                       'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 
                       'roll_mean_24', 'roll_std_24']
            
            # Scale data
            self.scaler = RobustScaler().fit(train[features])
            
            def scale(d): 
                return self.scaler.transform(d[features]), d['anomaly'].values
            
            X_train, y_train = scale(train)
            X_val, y_val = scale(val)
            X_test, y_test = scale(self.test_data)
            
            # Train models
            self.models['ffnn'] = self.train_ffnn(X_train, y_train, X_val, y_val)
            self.models['lstm'] = self.train_lstm(X_train, y_train, X_val, y_val)
            self.models['xgboost'] = self.train_xgboost(X_train, y_train)
            
            # Make predictions
            self.results['ffnn'] = self.models['ffnn'].predict(X_test).ravel()
            self.results['lstm'] = self.models['lstm'].predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).ravel()
            self.results['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
            
            # Classify anomaly types
            self.classify_anomaly_types()
            
            return True
            
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def preprocess_data(self, df):
        """Preprocess data"""
        # Filter for Solar data
        df = df[df['Source'] == 'Solar'].copy()
        
        # Fill missing values
        mean_prod = df['Production'].mean()
        df['Production'] = df['Production'].fillna(mean_prod)
        
        # Feature engineering
        df['hour'] = pd.to_datetime(df['StartHour'], format='%H:%M:%S', errors='coerce').dt.hour
        df['hour'] = df['hour'].fillna(df['hour'].mode()[0])
        
        df['monthName'] = df['monthName'].astype(str)
        df['month'] = pd.to_datetime(df['monthName'], format='%B', errors='coerce').dt.month
        df['month'] = df['month'].fillna(df['month'].mode()[0])
        
        df['dayName'] = df['dayName'].astype(str)
        df['dayofweek'] = pd.to_datetime(df['dayName'], format='%A', errors='coerce').dt.dayofweek
        df['dayofweek'] = df['dayofweek'].fillna(df['dayofweek'].mode()[0]).astype(int)
        
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
        
        # Rolling statistics
        for w in [3,6,24]:
            df[f'roll_mean_{w}'] = df['Production'].rolling(w).mean().shift(1)
            df[f'roll_std_{w}'] = df['Production'].rolling(w).std().shift(1)
        
        df = df.ffill().fillna(0)
        return df
    
    def inject_anomalies(self, df, frac=0.02, seed=42):
        """Inject anomalies"""
        np.random.seed(seed)
        df_processed = df.copy()
        labels = np.zeros(len(df_processed))
        
        idxs = np.random.choice(range(48, len(df_processed)-48), size=int(frac*len(df_processed)), replace=False)
        
        for i in idxs:
            if np.random.rand()>0.5:   # spike
                df_processed.iloc[i, df_processed.columns.get_loc('Production')] *= np.random.uniform(1.5,3.0)
            else:                      # drop
                df_processed.iloc[i, df_processed.columns.get_loc('Production')] *= np.random.uniform(0.0,0.5)
            labels[i]=1
        
        df_processed['anomaly']=labels
        return df_processed
    
    def train_ffnn(self, X_train, y_train, X_val, y_val):
        """Train FFNN model"""
        inp=layers.Input(shape=(X_train.shape[1],))
        x=layers.Dense(128,activation='relu')(inp)
        x=layers.Dropout(0.2)(x)
        x=layers.Dense(64,activation='relu')(x)
        out=layers.Dense(1,activation='sigmoid')(x)
        model=models.Model(inp,out)
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
        model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=40,batch_size=256,verbose=0)
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val):
        """Train LSTM model"""
        X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        inp=layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        x=layers.LSTM(64)(inp)
        x=layers.Dropout(0.2)(x)
        out=layers.Dense(1,activation='sigmoid')(x)
        model=models.Model(inp,out)
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
        model.fit(X_train_seq,y_train,validation_data=(X_val_seq,y_val),epochs=40,batch_size=256,verbose=0)
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def classify_anomaly_types(self):
        """Classify anomaly types based on production changes"""
        self.test_data = self.test_data.copy()
        
        # Calculate production changes
        self.test_data['production_change'] = self.test_data['Production'].pct_change()
        
        # Classify anomalies for each model
        for model_name, predictions in self.results.items():
            anomaly_mask = (predictions >= 0.5).astype(int)
            
            # Determine anomaly types
            anomaly_types = []
            for i, is_anomaly in enumerate(anomaly_mask):
                if is_anomaly:
                    change = self.test_data.iloc[i]['production_change']
                    if pd.isna(change):
                        anomaly_types.append('Unknown')
                    elif change > 0.5:  # More than 50% increase
                        anomaly_types.append('Sudden Rise')
                    elif change < -0.5:  # More than 50% decrease
                        anomaly_types.append('Sudden Fall')
                    elif change > 0.1:  # Moderate increase
                        anomaly_types.append('Gradual Rise')
                    elif change < -0.1:  # Moderate decrease
                        anomaly_types.append('Gradual Fall')
                    else:
                        anomaly_types.append('Fluctuation')
                else:
                    anomaly_types.append('Normal')
            
            self.test_data[f'{model_name}_anomaly_type'] = anomaly_types
            self.test_data[f'{model_name}_prediction'] = predictions

# Initialize API
api = AnomalyDetectionAPI()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/train')
def train_models():
    """Train models endpoint"""
    success = api.load_and_train_models()
    if success:
        return jsonify({'status': 'success', 'message': 'Models trained successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to train models'})

@app.route('/api/metrics')
def get_metrics():
    """Get model metrics"""
    if not api.results:
        return jsonify({'error': 'Models not trained yet'})
    
    y_true = api.test_data['anomaly'].values
    metrics = {}
    
    for model_name, predictions in api.results.items():
        y_pred = (predictions >= 0.5).astype(int)
        
        metrics[model_name] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, predictions))
        }
    
    return jsonify(metrics)

@app.route('/api/anomalies')
def get_anomalies():
    """Get anomaly data with types"""
    if api.test_data is None:
        return jsonify({'error': 'Models not trained yet'})
    
    # Prepare data for frontend
    data = []
    for idx, row in api.test_data.iterrows():
        data.append({
            'date': str(idx),
            'production': float(row['Production']),
            'true_anomaly': int(row['anomaly']),
            'ffnn_prediction': float(row.get('ffnn_prediction', 0)),
            'ffnn_anomaly_type': row.get('ffnn_anomaly_type', 'Normal'),
            'lstm_prediction': float(row.get('lstm_prediction', 0)),
            'lstm_anomaly_type': row.get('lstm_anomaly_type', 'Normal'),
            'xgboost_prediction': float(row.get('xgboost_prediction', 0)),
            'xgboost_anomaly_type': row.get('xgboost_anomaly_type', 'Normal')
        })
    
    return jsonify(data)

@app.route('/api/anomaly_types')
def get_anomaly_types():
    """Get anomaly type statistics"""
    if api.test_data is None:
        return jsonify({'error': 'Models not trained yet'})
    
    stats = {}
    for model_name in ['ffnn', 'lstm', 'xgboost']:
        type_col = f'{model_name}_anomaly_type'
        if type_col in api.test_data.columns:
            type_counts = api.test_data[type_col].value_counts().to_dict()
            stats[model_name] = type_counts
    
    return jsonify(stats)

@app.route('/api/plot/<plot_type>')
def get_plot(plot_type):
    """Generate and return plot as base64"""
    if api.test_data is None:
        return jsonify({'error': 'Models not trained yet'})
    
    plt.figure(figsize=(12, 8))
    
    if plot_type == 'time_series':
        # Combined time series plot with anomalies
        plt.plot(api.test_data.index, api.test_data['Production'], label='Production', alpha=0.7, color='blue', linewidth=1)
        
        # Plot anomalies for each model
        colors = ['red', 'orange', 'purple']
        model_names = ['ffnn', 'lstm', 'xgboost']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            pred_col = f'{model_name}_prediction'
            if pred_col in api.test_data.columns:
                anomalies = api.test_data[api.test_data[pred_col] >= 0.5]
                if not anomalies.empty:
                    plt.scatter(anomalies.index, anomalies['Production'], 
                              label=f'{model_name.upper()} Anomalies', 
                              color=color, s=30, alpha=0.7, marker='o')
        
        plt.title('Production Time Series with All Model Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Production')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    elif plot_type.startswith('time_series_'):
        # Individual model time series with anomaly types
        model_name = plot_type.replace('time_series_', '')
        if model_name in ['ffnn', 'lstm', 'xgboost']:
            plt.figure(figsize=(14, 8))
            
            # Plot production
            plt.plot(api.test_data.index, api.test_data['Production'], 
                    label='Production', alpha=0.7, color='blue', linewidth=1.5)
            
            # Get predictions and anomaly types
            pred_col = f'{model_name}_prediction'
            type_col = f'{model_name}_anomaly_type'
            
            if pred_col in api.test_data.columns and type_col in api.test_data.columns:
                anomalies = api.test_data[api.test_data[pred_col] >= 0.5]
                
                if not anomalies.empty:
                    # Plot different anomaly types with different colors and markers
                    anomaly_colors = {
                        'Sudden Rise': 'red',
                        'Sudden Fall': 'darkblue',
                        'Gradual Rise': 'orange',
                        'Gradual Fall': 'purple',
                        'Fluctuation': 'green'
                    }
                    
                    anomaly_markers = {
                        'Sudden Rise': '^',
                        'Sudden Fall': 'v',
                        'Gradual Rise': 's',
                        'Gradual Fall': 'D',
                        'Fluctuation': 'o'
                    }
                    
                    for anomaly_type, color in anomaly_colors.items():
                        type_anomalies = anomalies[anomalies[type_col] == anomaly_type]
                        if not type_anomalies.empty:
                            plt.scatter(type_anomalies.index, type_anomalies['Production'],
                                      label=f'{anomaly_type}', color=color, s=50, 
                                      marker=anomaly_markers.get(anomaly_type, 'o'), alpha=0.8)
            
            plt.title(f'{model_name.upper()} Model - Production Time Series with Anomaly Types')
            plt.xlabel('Date')
            plt.ylabel('Production')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
    elif plot_type == 'anomaly_types':
        # Anomaly types distribution
        stats = {}
        for model_name in ['ffnn', 'lstm', 'xgboost']:
            type_col = f'{model_name}_anomaly_type'
            if type_col in api.test_data.columns:
                # Filter out 'Normal' for anomaly type distribution
                anomalies_only = api.test_data[api.test_data[type_col] != 'Normal']
                if not anomalies_only.empty:
                    stats[model_name.upper()] = anomalies_only[type_col].value_counts()
        
        if stats:
            df_stats = pd.DataFrame(stats).fillna(0)
            df_stats.plot(kind='bar', figsize=(12, 6))
            plt.title('Anomaly Types Distribution by Model')
            plt.xlabel('Anomaly Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    elif plot_type == 'confusion_matrices':
        # Confusion matrices at 0.5 threshold with best model highlighted
        y_true = api.test_data['anomaly'].values
        
        # Find best model by AUC
        model_aucs = {}
        for model_name in ['ffnn', 'lstm', 'xgboost']:
            if model_name in api.results:
                model_aucs[model_name] = roc_auc_score(y_true, api.results[model_name])
        
        best_model = max(model_aucs.keys(), key=lambda x: model_aucs[x])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_names = ['ffnn', 'lstm', 'xgboost']
        colors = ['Blues', 'Oranges', 'Purples']
        
        for i, model_name in enumerate(model_names):
            if model_name in api.results:
                y_pred = (api.results[model_name] >= 0.5).astype(int)
                cm = confusion_matrix(y_true, y_pred)
                
                # Calculate metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, api.results[model_name])
                
                # Highlight best model
                if model_name == best_model:
                    cmap = 'Greens'
                    title_color = 'green'
                    title_prefix = '🏆 BEST (AUC: {:.3f})'.format(auc)
                else:
                    cmap = colors[i]
                    title_color = 'black'
                    title_prefix = 'AUC: {:.3f}'.format(auc)
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[i], 
                          cbar=False, square=True, linewidths=0.5)
                
                axes[i].set_title(f'{model_name.upper()}\n{title_prefix}\nAcc: {acc:.3f}, P: {prec:.3f}, R: {rec:.3f}, F1: {f1:.3f}',
                                fontweight='bold' if model_name == best_model else 'normal',
                                color=title_color)
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                axes[i].set_xticklabels(['Normal', 'Anomaly'])
                axes[i].set_yticklabels(['Normal', 'Anomaly'])
        
        plt.suptitle('Confusion Matrices at Threshold 0.5 (Best Model Highlighted)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
    elif plot_type == 'roc_curves':
        # ROC curves comparison
        y_true = api.test_data['anomaly'].values
        plt.figure(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        model_names = ['ffnn', 'lstm', 'xgboost']
        
        # Find best model by AUC
        model_aucs = {}
        for model_name in model_names:
            if model_name in api.results:
                model_aucs[model_name] = roc_auc_score(y_true, api.results[model_name])
        
        best_model = max(model_aucs.keys(), key=lambda x: model_aucs[x])
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            if model_name in api.results:
                fpr, tpr, _ = roc_curve(y_true, api.results[model_name])
                auc_score = roc_auc_score(y_true, api.results[model_name])
                
                # Highlight best model
                linewidth = 3 if model_name == best_model else 2
                alpha = 1.0 if model_name == best_model else 0.8
                
                plt.plot(fpr, tpr, color=color, linewidth=linewidth, alpha=alpha,
                        label=f'{model_name.upper()} (AUC = {auc_score:.3f})' + 
                              (' 🏆 BEST' if model_name == best_model else ''))
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison (Best Model Highlighted)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
    
    # Convert plot to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_data})

@app.route('/api/best_model')
def get_best_model():
    """Get best model recommendation based on AUC only"""
    if not api.results:
        return jsonify({'error': 'Models not trained yet'})
    
    y_true = api.test_data['anomaly'].values
    model_scores = {}
    
    for model_name, predictions in api.results.items():
        y_pred = (predictions >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, predictions)
        
        model_scores[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc': auc
        }
    
    # Find best model by AUC only
    best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['auc'])
    best_score = model_scores[best_model]
    
    # Generate reasoning based on AUC
    reasoning = f"""
    **{best_model.upper()}** is the best model based on AUC score:
    
    🎯 **AUC Score**: {best_score['auc']:.3f} (Highest among all models)
    
    📊 **AUC Interpretation**: {best_score['auc']:.1%} probability that the model ranks a random positive instance higher than a random negative instance
    
    💡 **Why AUC is the best metric**:
    - Threshold-independent evaluation
    - Measures overall discriminative ability
    - Robust to class imbalance
    - Standard for binary classification performance
    
    📈 **Additional Metrics**:
    - Accuracy: {best_score['accuracy']:.3f}
    - Precision: {best_score['precision']:.3f}
    - Recall: {best_score['recall']:.3f}
    - F1 Score: {best_score['f1_score']:.3f}
    """
    
    return jsonify({
        'best_model': best_model,
        'best_score': best_score,
        'all_scores': model_scores,
        'reasoning': reasoning.strip()
    })

if __name__ == '__main__':
    print("Starting Anomaly Detection Web Application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
