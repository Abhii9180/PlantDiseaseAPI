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

# -----------------------------------------------------------
# Human-readable labels
# -----------------------------------------------------------
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
# Treatment recommendations
# -----------------------------------------------------------
TREATMENT_INFO = {
    "Apple___Apple_scab": "Spray Mancozeb (0.25%) or Captan (0.3%) every 10–15 days. Prune and destroy affected leaves.",
    "Apple___Black_rot": "Apply Captan (0.3%) or Thiophanate-methyl (0.1%). Remove mummified fruits and infected twigs.",
    "Apple___Cedar_apple_rust": "Spray Mancozeb (0.25%) or Sulphur (0.2%). Remove nearby juniper trees if possible.",
    "Apple___healthy": "Your Apple plant looks healthy! Maintain regular watering and pruning.",

    "Blueberry___healthy": "Your Blueberry plant is healthy. Maintain soil pH between 4.5–5.5 and avoid waterlogging.",

    "Cherry_(including_sour)___Powdery_mildew": "Apply Wettable Sulphur (0.3%) or Hexaconazole (0.1%) every 10 days. Ensure good air circulation.",
    "Cherry_(including_sour)___healthy": "Your Cherry plant is healthy! Maintain spacing and proper sunlight.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Spray Mancozeb (0.25%) or Propiconazole (0.1%) at 10-day intervals.",
    "Corn_(maize)___Common_rust_": "Use Mancozeb (0.25%) or Hexaconazole (0.1%). Grow rust-resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply Carbendazim (0.1%) or Mancozeb (0.25%) and destroy crop residues.",
    "Corn_(maize)___healthy": "Corn is healthy! Keep field clean and monitor for rust symptoms.",

    "Grape___Black_rot": "Spray Mancozeb (0.25%) or Myclobutanil (0.05%) at early symptom appearance.",
    "Grape___Esca_(Black_Measles)": "No complete cure; prune and burn infected parts. Apply Copper oxychloride (0.3%) after pruning.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply Mancozeb (0.25%) or Chlorothalonil (0.2%). Maintain canopy aeration.",
    "Grape___healthy": "Your Grape plant is healthy. Maintain regular pruning and balanced fertilization.",

    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected plants. Control psyllid vectors using Imidacloprid (0.3 ml/L) or Thiamethoxam (0.25 g/L).",

    "Peach___Bacterial_spot": "Spray Copper oxychloride (0.3%) every 10 days. Avoid overhead irrigation.",
    "Peach___healthy": "Your Peach plant looks healthy! Regular pruning and balanced NPK fertilization recommended.",

    "Pepper,_bell___Bacterial_spot": "Apply Copper oxychloride (0.3%) + Streptocycline (0.3 g/L). Avoid overhead watering.",
    "Pepper,_bell___healthy": "Your Bell Pepper plant is healthy! Maintain good ventilation and spacing.",

    "Potato___Early_blight": "Spray Mancozeb (0.25%) or Chlorothalonil (0.2%) at 10-day intervals.",
    "Potato___Late_blight": "Use Metalaxyl + Mancozeb (0.25%) or Cymoxanil (0.3%) every 7 days.",
    "Potato___healthy": "Your Potato crop looks healthy! Keep monitoring during wet seasons.",

    "Raspberry___healthy": "Raspberry plant is healthy. Maintain mulch and regular pruning.",

    "Soybean___healthy": "Soybean is healthy. Apply Rhizobium inoculation and maintain proper drainage.",

    "Squash___Powdery_mildew": "Spray Wettable Sulphur (0.3%) or Hexaconazole (0.1%) every 10 days.",

    "Strawberry___Leaf_scorch": "Apply Captan (0.3%) or Mancozeb (0.25%) every 10 days. Remove infected leaves.",
    "Strawberry___healthy": "Your Strawberry plant looks healthy! Avoid overhead watering.",

    "Tomato___Bacterial_spot": "Spray Copper oxychloride (0.3%) or Streptocycline (0.3 g/L). Remove affected leaves.",
    "Tomato___Early_blight": "Use Mancozeb (0.25%) or Chlorothalonil (0.2%) at 10-day intervals.",
    "Tomato___Late_blight": "Apply Metalaxyl + Mancozeb (0.25%) or Cymoxanil (0.3%) every 7–10 days.",
    "Tomato___Leaf_Mold": "Spray Mancozeb (0.25%) or Chlorothalonil (0.2%) and maintain ventilation.",
    "Tomato___Septoria_leaf_spot": "Apply Copper fungicide (0.3%) or Mancozeb (0.25%) at weekly intervals.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray Abamectin (0.2 ml/L) or Fenpyroximate (0.5 ml/L). Maintain humidity.",
    "Tomato___Target_Spot": "Use Mancozeb (0.25%) or Chlorothalonil (0.2%). Rotate with fungicides.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants and control whiteflies with Imidacloprid (0.3 ml/L).",
    "Tomato___Tomato_mosaic_virus": "Remove affected plants. Disinfect tools using 10% bleach. Avoid tobacco contact.",
    "Tomato___healthy": "Your Tomato plant is healthy! Maintain 6–8 hours of sunlight and balanced fertilizer."
}

# -----------------------------------------------------------
# Initialize Flask App and Cached Model Loader
# -----------------------------------------------------------
app = Flask(__name__)

try:
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading global labels file: {e}")
    raise e

_model = None
def get_model():
    """Load model once and reuse."""
    global _model
    if _model is None:
        print("Loading model into memory...")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

# -----------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Accepts an image file via POST request, runs inference, and returns JSON."""
    try:
        model = get_model()
    except Exception as e:
        app.logger.error(f"Prediction failed: Could not load model: {e}")
        return jsonify({"error": "Internal server error: Model loading failed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream).convert("RGB").resize(IMAGE_SIZE)
        # ⚠️ No normalization — model trained on 0–255 values
        x = np.array(img, dtype="float32")[None, ...]

        preds = model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx])
        label = class_names[top_idx]
        human_label = PRETTY_LABELS.get(label, label)
        treatment = TREATMENT_INFO.get(label, "No treatment data available.")

        return jsonify({
            "predicted_class": human_label,
            "confidence": round(conf, 4),
            "raw_label": label,
            "treatment": treatment
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Online",
        "message": "Plant Disease Classifier API working!",
        "endpoint": "POST to /predict with image file under key 'file'"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
