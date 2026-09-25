from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
# import os removed, as it's not strictly needed for the functions used here

# Import core functions from the anomaly_detector module
from anomaly_detector import get_model_metrics, detect_anomalies_and_plot, generate_comparison_plot, get_inference_result

# --- Initial Plot Generation on Server Start ---
print("Attempting to generate plots on server startup...")
try:
    generate_comparison_plot()
    detect_anomalies_and_plot()
    print("Plot files successfully created in backend/plots/")
except Exception as e:
    # Simplified error logging to standard print, which should clear warnings
    print(f"CRITICAL ERROR: Initial plot generation failed. Check data/wind_solar.csv and dependencies. Error: {e}") 

# Flask setup
app = Flask(__name__, 
            static_folder='../frontend',
            template_folder='../frontend')
CORS(app) 

# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main frontend page."""
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Returns the model comparison metrics."""
    return jsonify(get_model_metrics())

@app.route('/api/anomalies', methods=['GET'])
def anomalies():
    """Runs anomaly detection and returns the results."""
    anomalies_data, _ = detect_anomalies_and_plot()
    if 'error' in anomalies_data:
        return jsonify(anomalies_data), 500
    return jsonify(anomalies_data)

@app.route('/api/inference', methods=['GET'])
def inference():
    """Returns the final inference conclusion."""
    return jsonify(get_inference_result())

@app.route('/plots/<filename>', methods=['GET'])
def serve_plot(filename):
    """Serves the generated plot images from the 'plots' subdirectory."""
    # This function handles the path correctly
    return send_from_directory('plots', filename)

if __name__ == '__main__':
    print("Flask App running at http://127.0.0.1:5000/")
    app.run(debug=True, use_reloader=False)