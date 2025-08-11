"""
Image Recognition Web App (single-file)
Technologies: Python, TensorFlow/Keras, OpenCV, Flask

How it works:
- Uses a pretrained MobileNetV2 (ImageNet) model from Keras to classify objects.
- Uses OpenCV to read and preprocess uploaded images.
- Flask provides a simple upload form and a /predict API that returns top-k predictions.

Run:
1) Create a virtual environment (recommended) and install:
   pip install tensorflow flask opencv-python-headless numpy

2) Start the app:
   python image_recognition_app.py

3) Open http://127.0.0.1:5000 in your browser and upload an image.

Notes:
- This is an example starter. For production, consider model caching, async workers,
  larger timeouts, secure file handling, and a lightweight model for faster inference.
"""

from io import BytesIO
import base64
import os
from typing import List, Tuple

import numpy as np
import cv2
from flask import Flask, request, render_template_string, jsonify

# Import Keras model utilities
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------- Configuration --------------------
MODEL = None
IMG_WIDTH = 224
IMG_HEIGHT = 224
TOP_K = 5
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# -------------------- Flask App --------------------
app = Flask(__name__)

# Basic HTML template (minimal)
INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Image Recognition Demo</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; padding: 2rem; }
      .container { max-width: 700px; margin: 0 auto; }
      .preview { max-width: 100%; margin-top: 1rem; }
      .preds { margin-top: 1rem; }
      table { border-collapse: collapse; width: 100%; }
      th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Image Recognition (MobileNetV2)</h1>
      <p>Upload an image and receive top-{{top_k}} predictions (ImageNet labels).</p>

      <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Upload & Predict</button>
      </form>

      {% if img_data %}
        <img src="data:image/jpeg;base64,{{img_data}}" class="preview" alt="uploaded image">
      {% endif %}

      {% if preds %}
        <div class="preds">
          <h3>Predictions</h3>
          <table>
            <thead><tr><th>Rank</th><th>Label</th><th>Probability</th></tr></thead>
            <tbody>
            {% for i, p in enumerate(preds, start=1) %}
              <tr><td>{{i}}</td><td>{{p[1]}}</td><td>{{"{:.2f}%".format(p[2]*100)}}</td></tr>
            {% endfor %}
            </tbody>
          </table>
        </div>
      {% endif %}

    </div>
  </body>
</html>
"""

# -------------------- Model utilities --------------------

def allowed_file(filename: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS


def load_model_once():
    global MODEL
    if MODEL is None:
        print("Loading MobileNetV2 model (ImageNet weights). This may take a moment...")
        MODEL = MobileNetV2(weights='imagenet')
        print("Model loaded.")
    return MODEL


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Read image bytes into an OpenCV BGR image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")
    return img


def preprocess_for_mobilenet(cv2_bgr_img: np.ndarray) -> np.ndarray:
    """Resize, convert to RGB, and run mobilenet preprocessing.

    Returns a batch of shape (1, H, W, 3)
    """
    # Convert BGR (OpenCV) -> RGB
    rgb = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2RGB)
    # Resize to model expected size
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    array = img_to_array(resized)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def predict_image(image_bytes: bytes, top_k: int = TOP_K) -> List[Tuple[str, str, float]]:
    """Return top-k ImageNet predictions as tuples (class_id, label, prob)."""
    model = load_model_once()
    img = read_image_from_bytes(image_bytes)
    x = preprocess_for_mobilenet(img)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top_k)[0]
    # decoded is list of (class_id, label, prob)
    return decoded


# -------------------- Flask Routes --------------------

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, preds=None, img_data=None, top_k=TOP_K)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template_string(INDEX_HTML, preds=None, img_data=None, top_k=TOP_K, error='No file part')

    f = request.files['image']
    if f.filename == '':
        return render_template_string(INDEX_HTML, preds=None, img_data=None, top_k=TOP_K, error='No file selected')

    if not allowed_file(f.filename):
        return render_template_string(INDEX_HTML, preds=None, img_data=None, top_k=TOP_K, error='File type not allowed')

    image_bytes = f.read()

    try:
        preds = predict_image(image_bytes, top_k=TOP_K)
    except Exception as e:
        return render_template_string(INDEX_HTML, preds=None, img_data=None, top_k=TOP_K, error=str(e))

    # Encode the uploaded image to base64 for display in the page
    try:
        # convert bytes to JPEG (if not) using OpenCV to ensure browser compatibility
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception:
        img_b64 = None

    # Render results page
    return render_template_string(INDEX_HTML, preds=preds, img_data=img_b64, top_k=TOP_K)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint. Accepts form-data file field 'image' OR raw binary body.

    Returns JSON with predictions: [{"class_id":..., "label":..., "prob":...}, ...]
    """
    if 'image' in request.files:
        f = request.files['image']
        image_bytes = f.read()
    else:
        # try raw binary
        image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({'error': 'no image provided'}), 400

    try:
        preds = predict_image(image_bytes, top_k=TOP_K)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result = [
        {"class_id": c, "label": lbl, "prob": float(prob)} for (c, lbl, prob) in preds
    ]
    return jsonify({"predictions": result})


if __name__ == '__main__':
    # Preload model before first request for faster first inference
    load_model_once()
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
