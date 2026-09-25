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
import matplotlib

# Use a non-GUI backend so Flask can generate plots safely in background threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
from werkzeug.utils import secure_filename

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

# Create directories
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AnomalyDetectionAPI:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.test_data = None
        self.results = {}
        self.uploaded_file = None
        self.original_data = None
        
    def load_and_preprocess_user_data(self, file_path):
        """Load and preprocess user-uploaded CSV data"""
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            self.original_data = df.copy()
            
            print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            
            # Validate required columns
            required_columns = ['Production', 'Source', 'StartHour', 'monthName', 'dayName']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            # Check if we have enough data
            if len(df) < 100:
                return False, f"Dataset too small. Need at least 100 rows, got {len(df)}."
            
            # Preprocess the data
            df_processed = self.preprocess_data(df)
            
            if df_processed.empty or len(df_processed) < 100:
                return False, "Data preprocessing failed or resulted in insufficient data."
            
            print(f"Preprocessed dataset has {len(df_processed)} rows")
            
            # Inject anomalies
            df_anomalies = self.inject_anomalies(df_processed)
            
            print(f"Dataset after anomaly injection: {len(df_anomalies)} rows")
            
            # Train/test split
            N = len(df_anomalies)
            train_size = max(50, int(0.7 * N))
            val_size = max(30, int(0.85 * N))
            
            train = df_anomalies.iloc[:train_size]
            val = df_anomalies.iloc[train_size:val_size]
            self.test_data = df_anomalies.iloc[val_size:]
            
            print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(self.test_data)}")
            
            # Features
            features = ['Production', 'hour', 'dayofweek', 'month', 'is_weekend',
                       'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 
                       'roll_mean_24', 'roll_std_24']
            
            # Check features exist
            missing_features = [f for f in features if f not in train.columns]
            if missing_features:
                return False, f"Missing features after preprocessing: {missing_features}"
            
            # Scale data
            self.scaler = RobustScaler().fit(train[features])
            
            def scale(d): 
                try:
                    return self.scaler.transform(d[features]), d['anomaly'].values
                except Exception as e:
                    print(f"Scaling error: {e}")
                    return None, None
            
            X_train, y_train = scale(train)
            X_val, y_val = scale(val)
            X_test, y_test = scale(self.test_data)
            
            if X_train is None or len(X_train) == 0:
                return False, "Failed to scale training data."
            
            return True, f"Data loaded and preprocessed successfully. Training set size: {len(X_train)}"
            
        except Exception as e:
            print(f"Error in load_and_preprocess_user_data: {e}")
            return False, f"Error processing data: {str(e)}"
    
    def train_models_on_user_data(self):
        """Train models on user-uploaded data"""
        try:
            if self.original_data is None or len(self.original_data) < 100:
                return False, "Dataset too small. Need at least 100 rows for training."
            
            # Recreate full dataset from original data
            df_processed = self.preprocess_data(self.original_data)
            
            if df_processed.empty or len(df_processed) < 100:
                return False, "Data preprocessing failed or resulted in insufficient data."
            
            # Inject anomalies
            df_anomalies = self.inject_anomalies(df_processed)
            
            if len(df_anomalies) < 100:
                return False, "Dataset too small after anomaly injection."
            
            # Train/test split
            N = len(df_anomalies)
            train_size = max(50, int(0.7 * N))
            val_size = max(30, int(0.85 * N))
            
            train = df_anomalies.iloc[:train_size]
            val = df_anomalies.iloc[train_size:val_size]
            test = df_anomalies.iloc[val_size:]
            
            # Ensure we have enough data for each split
            if len(train) < 50 or len(val) < 30 or len(test) < 20:
                return False, "Insufficient data for train/validation/test split."
            
            features = ['Production', 'hour', 'dayofweek', 'month', 'is_weekend',
                       'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 
                       'roll_mean_24', 'roll_std_24']
            
            # Check if all features exist
            missing_features = [f for f in features if f not in train.columns]
            if missing_features:
                return False, f"Missing features after preprocessing: {missing_features}"
            
            # Scale data
            self.scaler = RobustScaler().fit(train[features])
            
            def scale(d): 
                try:
                    return self.scaler.transform(d[features]), d['anomaly'].values
                except Exception as e:
                    print(f"Scaling error: {e}")
                    return None, None
            
            X_train, y_train = scale(train)
            X_val, y_val = scale(val)
            X_test, y_test = scale(test)
            
            if X_train is None or len(X_train) == 0:
                return False, "Failed to scale training data."
            
            # Train models
            print(f"Training models with {len(X_train)} samples...")
            self.models['ffnn'] = self.train_ffnn(X_train, y_train, X_val, y_val)
            self.models['lstm'] = self.train_lstm(X_train, y_train, X_val, y_val)
            self.models['xgboost'] = self.train_xgboost(X_train, y_train)
            
            # Make predictions
            self.results['ffnn'] = self.models['ffnn'].predict(X_test).ravel()
            self.results['lstm'] = self.models['lstm'].predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).ravel()
            self.results['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
            
            # Update test data with predictions
            self.test_data = test.copy()
            self.test_data['true_anomaly'] = y_test
            
            # Classify anomaly types
            self.classify_anomaly_types()
            
            return True, "Models trained successfully"
            
        except Exception as e:
            print(f"Training error: {e}")
            return False, f"Error training models: {str(e)}"
    
    def preprocess_data(self, df):
        """Preprocess user-uploaded data"""
        try:
            # Filter for Solar data (or use all if no Source column)
            if 'Source' in df.columns:
                df = df[df['Source'] == 'Solar'].copy()
            
            # Fill missing Production with mean
            if 'Production' in df.columns:
                mean_prod = df['Production'].mean()
                df['Production'] = df['Production'].fillna(mean_prod)
            else:
                return pd.DataFrame()  # Return empty if no Production column
            
            # Feature engineering - handle time columns
            if 'StartHour' in df.columns:
                try:
                    df['hour'] = pd.to_datetime(df['StartHour'], format='%H:%M:%S', errors='coerce').dt.hour
                    df['hour'] = df['hour'].fillna(df['hour'].mode()[0])
                except:
                    df['hour'] = 12  # Default to noon if parsing fails
            else:
                df['hour'] = 12  # Default hour
            
            # Process month
            if 'monthName' in df.columns:
                try:
                    df['monthName'] = df['monthName'].astype(str)
                    df['month'] = pd.to_datetime(df['monthName'], format='%B', errors='coerce').dt.month
                    df['month'] = df['month'].fillna(df['month'].mode()[0])
                except:
                    df['month'] = 6  # Default to June
            else:
                df['month'] = 6  # Default month
            
            # Process day of week
            if 'dayName' in df.columns:
                try:
                    df['dayName'] = df['dayName'].astype(str)
                    df['dayofweek'] = pd.to_datetime(df['dayName'], format='%A', errors='coerce').dt.dayofweek
                    df['dayofweek'] = df['dayofweek'].fillna(df['dayofweek'].mode()[0]).astype(int)
                except:
                    df['dayofweek'] = 1  # Default to Monday
            else:
                df['dayofweek'] = 1  # Default day
            
            df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
            
            # Rolling statistics
            for w in [3,6,24]:
                if len(df) > w:
                    df[f'roll_mean_{w}'] = df['Production'].rolling(w).mean().shift(1)
                    df[f'roll_std_{w}'] = df['Production'].rolling(w).std().shift(1)
                else:
                    df[f'roll_mean_{w}'] = df['Production'].mean()
                    df[f'roll_std_{w}'] = df['Production'].std()
            
            df = df.ffill().fillna(0)
            return df
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return pd.DataFrame()
    
    def inject_anomalies(self, df, frac=0.02, seed=42):
        """Inject anomalies into user data"""
        np.random.seed(seed)
        df_processed = df.copy()
        labels = np.zeros(len(df_processed))
        
        # Adjust injection based on data size
        min_size = max(100, len(df_processed) // 10)
        if len(df_processed) < min_size * 2:
            min_size = len(df_processed) // 4
        
        idxs = np.random.choice(range(min_size, len(df_processed)-min_size), 
                              size=max(10, int(frac*len(df_processed))), replace=False)
        
        for i in idxs:
            if i < len(df_processed):
                if np.random.rand()>0.5:   # spike
                    df_processed.iloc[i, df_processed.columns.get_loc('Production')] *= np.random.uniform(1.5,3.0)
                else:                      # drop
                    df_processed.iloc[i, df_processed.columns.get_loc('Production')] *= np.random.uniform(0.0,0.5)
                labels[i]=1
        
        df_processed['anomaly']=labels
        return df_processed
    
    def train_ffnn(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
        """Train FFNN model"""
        inp=layers.Input(shape=(X_train.shape[1],))
        x=layers.Dense(128,activation='relu')(inp)
        x=layers.Dropout(0.2)(x)
        x=layers.Dense(64,activation='relu')(x)
        out=layers.Dense(1,activation='sigmoid')(x)
        model=models.Model(inp,out)
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
        model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=batch_size,verbose=0)
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=256):
        """Train LSTM model"""
        X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        inp=layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        x=layers.LSTM(64)(inp)
        x=layers.Dropout(0.2)(x)
        out=layers.Dense(1,activation='sigmoid')(x)
        model=models.Model(inp,out)
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
        model.fit(X_train_seq,y_train,validation_data=(X_val_seq,y_val),epochs=epochs,batch_size=batch_size,verbose=0)
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
            anomaly_reasons = []
            for i, is_anomaly in enumerate(anomaly_mask):
                if is_anomaly and i < len(self.test_data):
                    change = self.test_data.iloc[i]['production_change']
                    if pd.isna(change):
                        anomaly_types.append('Unknown')
                        anomaly_reasons.append('Change could not be computed (missing previous production value).')
                    elif change > 0.5:  # More than 50% increase
                        anomaly_types.append('Sudden Rise')
                        anomaly_reasons.append(f'Production increased by {change*100:.1f}% compared to previous point (>50%), classified as Sudden Rise (Spike).')
                    elif change < -0.5:  # More than 50% decrease
                        anomaly_types.append('Sudden Fall')
                        anomaly_reasons.append(f'Production decreased by {abs(change)*100:.1f}% compared to previous point (>50%), classified as Sudden Fall (Drop).')
                    elif change > 0.1:  # Moderate increase
                        anomaly_types.append('Gradual Rise')
                        anomaly_reasons.append(f'Production increased by {change*100:.1f}% compared to previous point (between 10% and 50%), classified as Gradual Rise.')
                    elif change < -0.1:  # Moderate decrease
                        anomaly_types.append('Gradual Fall')
                        anomaly_reasons.append(f'Production decreased by {abs(change)*100:.1f}% compared to previous point (between 10% and 50%), classified as Gradual Fall.')
                    else:
                        anomaly_types.append('Fluctuation')
                        anomaly_reasons.append(f'Production changed by {change*100:.1f}% (within ±10%), but the model still flagged this as an anomaly, classified as Fluctuation.')
                else:
                    anomaly_types.append('Normal')
                    anomaly_reasons.append('Not flagged as anomaly by the model at this point (prediction < 0.5).')
            
            self.test_data[f'{model_name}_anomaly_type'] = anomaly_types
            self.test_data[f'{model_name}_prediction'] = predictions
            self.test_data[f'{model_name}_reason'] = anomaly_reasons

# Initialize API
api = AnomalyDetectionAPI()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index_upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess the data
        success, message = api.load_and_preprocess_user_data(filepath)
        
        if success:
            api.uploaded_file = filename
            # Get data info
            df = api.original_data
            data_info = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'production_stats': {
                    'mean': float(df['Production'].mean()) if 'Production' in df.columns else 0,
                    'std': float(df['Production'].std()) if 'Production' in df.columns else 0,
                    'min': float(df['Production'].min()) if 'Production' in df.columns else 0,
                    'max': float(df['Production'].max()) if 'Production' in df.columns else 0
                }
            }
            return jsonify({
                'status': 'success', 
                'message': message,
                'data_info': data_info
            })
        else:
            return jsonify({'status': 'error', 'message': message})
    
    return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload a CSV file.'})

@app.route('/api/train')
def train_models():
    """Train models on uploaded data"""
    if not api.uploaded_file:
        return jsonify({'status': 'error', 'message': 'Please upload a dataset first'})
    
    success, message = api.train_models_on_user_data()
    
    if success:
        return jsonify({'status': 'success', 'message': message})
    else:
        return jsonify({'status': 'error', 'message': message})

@app.route('/api/metrics')
def get_metrics():
    """Get model metrics"""
    if not api.results:
        return jsonify({'error': 'Models not trained yet'})
    
    y_true = api.test_data['true_anomaly'].values
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
            'true_anomaly': int(row['true_anomaly']),
            'ffnn_prediction': float(row.get('ffnn_prediction', 0)),
            'ffnn_anomaly_type': row.get('ffnn_anomaly_type', 'Normal'),
            'ffnn_reason': row.get('ffnn_reason', 'Not flagged as anomaly by FFNN (prediction < 0.5).'),
            'lstm_prediction': float(row.get('lstm_prediction', 0)),
            'lstm_anomaly_type': row.get('lstm_anomaly_type', 'Normal'),
            'lstm_reason': row.get('lstm_reason', 'Not flagged as anomaly by LSTM (prediction < 0.5).'),
            'xgboost_prediction': float(row.get('xgboost_prediction', 0)),
            'xgboost_anomaly_type': row.get('xgboost_anomaly_type', 'Normal'),
            'xgboost_reason': row.get('xgboost_reason', 'Not flagged as anomaly by XGBoost (prediction < 0.5).')
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

@app.route('/api/best_model')
def get_best_model():
    """Get best model recommendation based on AUC only"""
    if not api.results:
        return jsonify({'error': 'Models not trained yet'})
    
    y_true = api.test_data['true_anomaly'].values
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
        
        plt.title(f'Production Time Series with All Model Anomalies\nDataset: {api.uploaded_file}')
        plt.xlabel('Data Point Index')
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
            
            plt.title(f'{model_name.upper()} Model - Production Time Series with Anomaly Types\nDataset: {api.uploaded_file}')
            plt.xlabel('Data Point Index')
            plt.ylabel('Production')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
    elif plot_type == 'confusion_matrices':
        # Confusion matrices at 0.5 threshold with best model highlighted
        y_true = api.test_data['true_anomaly'].values
        
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
        
        plt.suptitle(f'Confusion Matrices at Threshold 0.5 (Best Model Highlighted)\nDataset: {api.uploaded_file}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
    elif plot_type == 'roc_curves':
        # ROC curves comparison
        y_true = api.test_data['true_anomaly'].values
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
        plt.title(f'ROC Curves Comparison (Best Model Highlighted)\nDataset: {api.uploaded_file}', fontsize=14, fontweight='bold')
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

if __name__ == '__main__':
    print("Starting Anomaly Detection Web Application with File Upload...")
    app.run(debug=True, host='0.0.0.0', port=5000)
