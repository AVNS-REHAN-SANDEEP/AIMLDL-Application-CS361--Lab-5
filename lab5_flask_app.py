"""
RGIPT - CS361 AIMLDL Application
Lab Assignment 5 : Flask alternative to lab5_streamlit_app.py

RUN WITH:
    python lab5_flask_app.py
Then open http://127.0.0.1:5000 in your browser.

Single-file app (no templates/ folder needed) -- HTML is inlined for
simplicity, which is fine for a lab submission. Everything else mirrors the
Streamlit version: load best_model.keras + class_names.json, preprocess the
uploaded image to 128x128x3 in [0,1], predict, return class + confidence +
top-3.
"""

import json
import io
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf

MODEL_PATH = "outputs_lab4/best_model.keras"
CLASS_NAMES_PATH = "outputs_lab4/class_names.json"
IMG_SIZE = 128

app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

PAGE = """
<!doctype html>
<html>
<head>
  <title>Clothing Classifier</title>
  <style>
    body { font-family: sans-serif; max-width: 640px; margin: 40px auto; }
    .result { padding: 16px; border: 1px solid #ddd; border-radius: 8px; margin-top: 20px; }
    .bar { background: #4a90d9; height: 18px; margin: 4px 0; border-radius: 4px; }
    .row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
    .label { width: 140px; }
  </style>
</head>
<body>
  <h2>Clothing Image Classifier</h2>
  <p>Lab Assignment 5 -- CNN trained in Lab 4, deployed with Flask</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Predict</button>
  </form>
  {% if prediction %}
  <div class="result">
    <h3>Prediction: {{ prediction }} ({{ confidence }}%)</h3>
    <h4>Top-3</h4>
    {% for label, score in top3 %}
      <div class="row">
        <div class="label">{{ label }}</div>
        <div class="bar" style="width: {{ score }}%"></div>
        <div>{{ score }}%</div>
      </div>
    {% endfor %}
  </div>
  {% endif %}
</body>
</html>
"""


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, top3 = None, None, []
    if request.method == "POST":
        file = request.files["image"]
        x = preprocess(file.read())
        probs = model.predict(x, verbose=0)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        prediction = CLASS_NAMES[top3_idx[0]]
        confidence = round(float(probs[top3_idx[0]]) * 100, 2)
        top3 = [(CLASS_NAMES[i], round(float(probs[i]) * 100, 2)) for i in top3_idx]
    return render_template_string(PAGE, prediction=prediction, confidence=confidence, top3=top3)


if __name__ == "__main__":
    app.run(debug=True)
