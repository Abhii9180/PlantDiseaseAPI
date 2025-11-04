import os, json, time, uuid, numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf
import os

# --- All your existing logic remains unchanged ---

MODEL_FP  = "plant_mnv2.keras"
LABELS_FP = "labels.json"
IMAGE_SIZE = (160, 160)
UNCERTAIN_THRESHOLD = 0.60  # tweak later

# --- Load model + labels ---
model = tf.keras.models.load_model(MODEL_FP)
with open(LABELS_FP) as f:
    class_names = json.load(f)

# Optional pretty names for UI (fallbacks to raw name if missing)
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
    "Tomato___healthy": "Tomato — Healthy",
}

def prettify(name: str) -> str:
    return pretty.get(name, name)

def predict(img: Image.Image, allow_save: bool):
    if img is None:
        # Update the status text and return a "waiting" message for the label
        return {"Waiting for prediction...": 0.0}, "Please upload an image.", gr.update(visible=False, value="")

    # Prepare image -> model
    arr = np.array(img.convert("RGB").resize(IMAGE_SIZE), dtype="float32")[None, ...]
    probs = model.predict(arr, verbose=0)[0]  # shape (num_classes,)
    top_idx = probs.argsort()[-5:][::-1]
    top = [(class_names[i], float(probs[i])) for i in top_idx]

    # Build display dict for Label
    display = {prettify(name): conf for name, conf in top}

    # Uncertain flag
    primary_name, primary_conf = top[0]
    uncertain = primary_conf < UNCERTAIN_THRESHOLD

    # Optional logging for active learning
    note_text = ""
    if allow_save:
        os.makedirs("logs/images", exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        uid = uuid.uuid4().hex[:8]
        img_path = f"logs/images/{stamp}_{uid}.png"
        meta_path = f"logs/images/{stamp}_{uid}.json"
        img.convert("RGB").save(img_path, format="PNG")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "top5": [(name, float(conf)) for name, conf in top],
                    "uncertain": uncertain,
                    "threshold": UNCERTAIN_THRESHOLD,
                    "image_size": img.size,
                    "saved_at": stamp,
                },
                f,
                indent=2,
            )
        note_text = f"Saved a copy for future fine-tuning: **{os.path.basename(img_path)}**"

    # Guidance text
    if uncertain:
        status = (
            f"⚠️ **Prediction is uncertain** (Top confidence: {primary_conf:.1%}). "
            "Try a clearer photo with good lighting and a plain background."
        )
    else:
        status = f"✅ **Most likely:** {prettify(primary_name)} ({primary_conf:.1%})"

    # Return visibility update for the note markdown
    note_component_update = (
        gr.update(visible=True, value=note_text) if note_text else gr.update(visible=False, value="")
    )
    return display, status, note_component_update

# --- NEW UI DEFINITION ---

# 1. Define Custom CSS for styling
custom_css = """
/* --- Titles & Headers --- */
h1.title-md {
    color: #2E8B57; /* SeaGreen */
    text-align: center;
    font-size: 2.8rem !important;
    font-weight: 700;
    padding-bottom: 0;
    margin-bottom: 0;
}
p.subtitle-md {
    text-align: center;
    font-size: 1.1rem;
    color: #333;
    margin-top: 5px;
    margin-bottom: 25px;
}
h3.column-header {
    color: #1a5632; /* Darker green for step headers */
    font-size: 1.5rem !important;
    font-weight: 600;
    border-bottom: 2px solid #2E8B57;
    padding-bottom: 5px;
}

/* --- Button --- */
.predict-btn {
    min-height: 55px;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    border-radius: 10px !important;
}

/* --- Output Styling --- */
#status-message p {
    font-size: 1.2rem;
    font-weight: 500;
    padding: 1rem;
    border-radius: 10px;
    background-color: #f8fdf8; /* Very light green */
    border: 1px solid #d4e9d4;
    text-align: center;
    line-height: 1.5;
}

/* === CHANGE HERE: Replaced #output-box with .output-box-class === */
.output-box-class {
    border-radius: 10px !important;
    border: 2px solid #2E8B57;
    background-color: #ffffff;
    padding: 1rem !important;
}
#save-note p {
    font-style: italic;
    color: #444;
    background-color: #f9f9f9;
    padding: 0.75rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #eee;
}
"""

# 2. Create the Gradio Blocks UI
with gr.Blocks(
    title="🌿 Plant Disease Classifier",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime"), # Use a green theme
    css=custom_css
) as demo:

    # --- Title Section ---
    gr.Markdown(
        "<h1 class='title-md'>🌿 Plant Disease Classifier 🌿</h1>"
        "<p class='subtitle-md'>"
        "Upload a leaf photo to get predictions from a 38-class MobileNetV2 model."
        "</p>",
    )

    # --- Main Layout (2 columns) ---
    with gr.Row(variant="panel", equal_height=False):

        # --- LEFT COLUMN (INPUTS) ---
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("<h3 class='column-header'>Step 1: Upload Image</h3>")
            inp = gr.Image(type="pil", label="Upload leaf image", height=350)
            
            gr.Markdown("<h3 class='column-header'>Step 2: Settings</h3>")
            allow_save = gr.Checkbox(
                label="Allow saving this image to improve the model",
                value=False,
                info="Help us fine-tune! Check this to anonymously save your image for future training."
            )
            
            gr.Markdown("<h3 class='column-header'>Step 3: Classify!</h3>")
            btn = gr.Button("🔍 Predict Disease", variant="primary", elem_classes=["predict-btn"])

        # --- RIGHT COLUMN (OUTPUTS) ---
        with gr.Column(scale=2, min_width=500):
            gr.Markdown("<h3 class='column-header'>Results</h3>")
            
            # Status message (e.g., "Most likely: ...")
            out_status = gr.Markdown(
                value="Waiting for image... ⏳",
                elem_id="status-message"
            )
            
            # === CHANGE HERE: Replaced gr.Box with gr.Group ===
            # Group for the top-5 predictions
            with gr.Group(elem_classes=["output-box-class"]):
                gr.Markdown("#### Top-5 Predictions")
                out_label = gr.Label(
                    num_top_classes=5,
                    label="Predictions",
                    show_label=False # Hide default label, we have a custom header
                )
            
            # Note for when an image is saved
            out_note = gr.Markdown(visible=False, elem_id="save-note")

    # --- Click Handler (Unchanged) ---
    btn.click(fn=predict, inputs=[inp, allow_save], outputs=[out_label, out_status, out_note])

# --- Launch (Unchanged) ---
if __name__ == "__main__":
    # Use PORT from environment variable, default to 7860 if not set
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)
