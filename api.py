from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os

# -----------------------------------------------------------
# Load model and labels
# -----------------------------------------------------------
MODEL_PATH = "plant_mnv2.keras"
LABELS_PATH = "labels.json"
IMAGE_SIZE = (160, 160)

app = Flask(__name__)

# Load once on startup
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_names = json.load(f)

pretty = {
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
# API endpoint
# -----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file).convert("RGB").resize(IMAGE_SIZE)
        x = np.array(img, dtype="float32")[None, ...] / 255.0
        preds = model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx])
        label = class_names[top_idx]
        human = pretty.get(label, label)

        return jsonify({
            "predicted_class": human,
            "confidence": round(conf, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Plant Disease Classifier API working!"})


# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
