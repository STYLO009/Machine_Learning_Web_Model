import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "media_detector.keras")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = preprocess_image(img)
    pred = model.predict(img)[0][0]

    if pred < 0.49:
        result = "FAKE (AI Generated)"
        confidence = pred
    else:
        result = "REAL"
        confidence = 1 - pred

    return jsonify({
        "result": result,
        "confidence": f"{confidence*100:.2f}%"
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
