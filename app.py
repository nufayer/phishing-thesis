import os
import requests
from flask import Flask, render_template, request, jsonify

from predictor import predict_url
from feature_extractor import extract_features

app = Flask(__name__)

# ============================================================
# MODEL DOWNLOAD (runs at startup on Render free tier)
# ============================================================
MODEL_URL = os.environ.get("MODEL_URL")
MODEL_PATH = "models/ensemble_final.pkl"

def download_model():
    """Download model from MODEL_URL if not present"""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return True
    
    if not MODEL_URL:
        print("WARNING: MODEL_URL not set. Skipping download.")
        return False
    
    try:
        print(f"Downloading model from {MODEL_URL}...")
        os.makedirs("models", exist_ok=True)
        
        response = requests.get(MODEL_URL, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded to {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")
        return True
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        return False

# Download model at startup (only needed on free tier with ephemeral filesystem)
download_model()

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

@app.route("/health")
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}), 200

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "").strip()
    
    if not url:
        return render_template("index.html", error="Please enter a URL")
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        result = predict_url(url)
        features = extract_features(url)

        prediction = result["prediction"]
        confidence = result["confidence"]

        if prediction == "Legitimate":
            result_class = "safe"
            risk_score = round(100 - confidence, 2)
        else:
            result_class = "danger"
            risk_score = confidence

        return render_template(
            "index.html",
            url=url,
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            result_class=result_class,
            features=features
        )

    except Exception as e:
        return render_template("index.html", error=f"Analysis failed: {str(e)}")

# API endpoint for programmatic access
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL required"}), 400
    
    url = data["url"]
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        result = predict_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)