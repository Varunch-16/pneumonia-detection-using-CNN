import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "new.h5"
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
UPLOAD_FOLDER.mkdir(exist_ok=True)

# compile=False avoids loading old optimizer state from the saved .h5 file.
model = load_model(MODEL_PATH, compile=False)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(img_path: Path) -> np.ndarray:
    """Load and prepare an X-ray image for the trained EfficientNetB0 model."""
    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img)

    # The original training code used EfficientNetB0 and passed images as RGB arrays.
    # Do not divide by 255 here unless the model is retrained with that preprocessing.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("Please upload an X-ray image.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected. Please choose a PNG, JPG, or JPEG image.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PNG, JPG, or JPEG image.")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}_{safe_name}"
    file_path = app.config["UPLOAD_FOLDER"] / unique_name
    file.save(file_path)

    try:
        img_array = prepare_image(file_path)
        prediction_scores = model.predict(img_array)
        predicted_index = int(np.argmax(prediction_scores, axis=1)[0])
        prediction = CLASS_NAMES[predicted_index]
        confidence = float(np.max(prediction_scores) * 100)
    finally:
        if file_path.exists():
            file_path.unlink()

    return render_template("result.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
