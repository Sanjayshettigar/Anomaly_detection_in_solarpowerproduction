#!/usr/bin/env python3
"""Enhanced anomaly detection script with multi-threshold analysis."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anomaly_detector import (
    detect_anomalies_and_plot, 
    plot_confusion_matrices_multiple_thresholds,
    generate_comprehensive_metrics_comparison,
    load_data,
    clean_and_feature_engineer,
    inject_anomalies,
    train_and_evaluate_models
)

def main():
    print("🚀 Starting Enhanced Anomaly Detection Analysis...")
    
    # Run the basic detection and plotting
    print("📊 Running basic anomaly detection...")
    result, error = detect_anomalies_and_plot()
    
    if error:
        print(f"❌ Error in basic detection: {error}")
        return
    
    if result.get('status') != 'success':
        print(f"❌ Basic detection failed: {result}")
        return
    
    print("✅ Basic detection completed successfully!")
    
    # Get the data and predictions for enhanced analysis
    print("🔧 Setting up enhanced analysis...")
    df, error = load_data()
    if error:
        print(f"❌ Error loading data: {error}")
        return
    
    df = clean_and_feature_engineer(df)
    df = inject_anomalies(df, frac=0.02)
    test_results, scaler = train_and_evaluate_models(df)
    
    # Prepare predictions
    per_model_preds = {
        'FFNN': test_results['FFNN_pred'],
        'LSTM': test_results['LSTM_pred'],
        'Meta-Ensemble (XGBoost)': test_results['XGBoost_pred']
    }
    
    # Generate multi-threshold confusion matrices
    print("📈 Generating multi-threshold confusion matrices...")
    try:
        plot_confusion_matrices_multiple_thresholds(test_results, per_model_preds, thresholds=[0.5, 0.8, 0.35])
        print("✅ Multi-threshold confusion matrices generated!")
    except Exception as e:
        print(f"❌ Error generating multi-threshold confusion matrices: {e}")
    
    # Generate comprehensive metrics comparison
    print("📊 Generating comprehensive metrics comparison...")
    try:
        comparison_result = generate_comprehensive_metrics_comparison(test_results, per_model_preds, thresholds=[0.5, 0.8, 0.35])
        print("✅ Comprehensive metrics comparison generated!")
        
        # Print best model analysis
        best_analysis = comparison_result['best_model_analysis']
        print("\n🏆 MODEL PERFORMANCE ANALYSIS:")
        print("=" * 50)
        print(f"🥇 Best Model: {best_analysis['best_model']}")
        print(f"📈 Composite Score: {best_analysis['composite_score']}")
        print(f"🎯 AUC Score: {best_analysis['best_auc']}")
        print(f"💡 Recommendation: {best_analysis['recommendation']}")
        
        print("\n📋 Complete Model Rankings:")
        print("-" * 30)
        for model, metrics in best_analysis['model_rankings'].items():
            print(f"{model}:")
            print(f"  Accuracy: {metrics['Accuracy']}")
            print(f"  Precision: {metrics['Precision']}")
            print(f"  Recall: {metrics['Recall']}")
            print(f"  F1 Score: {metrics['F1 Score']}")
            print(f"  AUC: {metrics['AUC']}")
            print(f"  Composite Score: {metrics['Composite_Score']}")
            print()
        
    except Exception as e:
        print(f"❌ Error generating comprehensive comparison: {e}")
    
    print("\n🎉 Enhanced analysis completed! Check the plots directory for all generated visualizations.")

if __name__ == "__main__":
    main()
