import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
MODEL_PATH = "plant_mnv2.keras"
LABELS_PATH = "labels.json"
IMAGE_SIZE = (160, 160)

# Dictionary to map machine-readable labels to human-readable names
PRETTY_LABELS = {
    "Apple___Apple_scab": "Apple — Apple scab",
    "Apple___Black_rot": "Apple — Black rot",
    "Apple___Cedar_apple_rust": "Apple — Cedar apple rust",
    "Apple___healthy": "Apple — Healthy",
    "Blueberry___healthy": "Blueberry — Healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry (sour) — Powdery mildew",
    "Cherry_(including_sour)___healthy": "Cherry (sour) — Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn — Gray leaf spot",
    "Corn_(maize)___Common_rust_": "Corn — Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn — Northern leaf blight",
    "Corn_(maize)___healthy": "Corn — Healthy",
    "Grape___Black_rot": "Grape — Black rot",
    "Grape___Esca_(Black_Measles)": "Grape — Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape — Leaf blight (Isariopsis)",
    "Grape___healthy": "Grape — Healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange — Citrus greening",
    "Peach___Bacterial_spot": "Peach — Bacterial spot",
    "Peach___healthy": "Peach — Healthy",
    "Pepper,_bell___Bacterial_spot": "Bell Pepper — Bacterial spot",
    "Pepper,_bell___healthy": "Bell Pepper — Healthy",
    "Potato___Early_blight": "Potato — Early blight",
    "Potato___Late_blight": "Potato — Late blight",
    "Potato___healthy": "Potato — Healthy",
    "Raspberry___healthy": "Raspberry — Healthy",
    "Soybean___healthy": "Soybean — Healthy",
    "Squash___Powdery_mildew": "Squash — Powdery mildew",
    "Strawberry___Leaf_scorch": "Strawberry — Leaf scorch",
    "Strawberry___healthy": "Strawberry — Healthy",
    "Tomato___Bacterial_spot": "Tomato — Bacterial spot",
    "Tomato___Early_blight": "Tomato — Early blight",
    "Tomato___Late_blight": "Tomato — Late blight",
    "Tomato___Leaf_Mold": "Tomato — Leaf mold",
    "Tomato___Septoria_leaf_spot": "Tomato — Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato — Two-spotted spider mite",
    "Tomato___Target_Spot": "Tomato — Target spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato — Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato — Mosaic virus",
    "Tomato___healthy": "Tomato — Healthy"
}


# -----------------------------------------------------------
# Initialize Flask App and Load Global Resources
# -----------------------------------------------------------

app = Flask(__name__)

# Load model and labels once when the server starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    print("Model and labels loaded successfully.")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    # In a production environment, you might want the service to exit if this fails
    # raise e

# -----------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """Accepts an image file via POST request, runs inference, and returns JSON."""
    
    # 1. Input validation
    if "file" not in request.files:
        return jsonify({"error": "No 'file' part in the request. Please upload an image under the key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # 2. Preprocessing
        # Open file stream, convert to RGB, and resize to the expected input dimensions
        img = Image.open(file.stream).convert("RGB").resize(IMAGE_SIZE)
        
        # Convert image to numpy array, normalize, and add batch dimension
        x = np.array(img, dtype="float32")[None, ...] / 255.0
        
        # 3. Prediction
        preds = model.predict(x, verbose=0)[0]
        
        # 4. Result formatting
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx])
        
        # Get machine label and human-readable label
        label = class_names[top_idx]
        human_label = PRETTY_LABELS.get(label, label)

        # 5. Return JSON response
        return jsonify({
            "predicted_class": human_label,
            "confidence": round(conf, 4), # Use 4 decimal places for better precision
            "raw_label": label
        })
        
    except Exception as e:
        # Log the error for debugging on Render
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def home():
    """Simple health check endpoint."""
    return jsonify({
        "status": "Online",
        "message": "Plant Disease Classifier API working!",
        "endpoint": "POST to /predict with image file under key 'file'"
    })


# -----------------------------------------------------------
# Local/Gunicorn Deployment Configuration
# -----------------------------------------------------------

# IMPORTANT: When deploying with Gunicorn (Start Command: gunicorn api:app), 
# Gunicorn handles the host and port binding. The app.run() block below 
# is primarily for local testing and will be ignored by Gunicorn.
if __name__ == "__main__":
    # Get port from environment or default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)