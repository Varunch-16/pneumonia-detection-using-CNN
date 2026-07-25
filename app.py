from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from pathlib import Path
import numpy as np
import csv
import os

app = Flask(__name__)
app.secret_key = "replace-this-secret-key"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "new.h5"
FEEDBACK_FILE = BASE_DIR / "feedback.csv"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
UPLOAD_FOLDER.mkdir(exist_ok=True)

model = load_model(MODEL_PATH, compile=False)
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

DOCTORS = [
    {"hospital": "Cleveland Clinic", "doctor": "Pulmonology Department", "specialty": "Pulmonology", "maps_query": "Cleveland Clinic pulmonologist near me"},
    {"hospital": "University Hospitals", "doctor": "Respiratory Care / Pulmonology", "specialty": "Pulmonology", "maps_query": "University Hospitals pulmonologist near me"},
    {"hospital": "Nearby Specialist", "doctor": "Pulmonologist Near Me", "specialty": "Lung / Respiratory Specialist", "maps_query": "pulmonologist near me"},
]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("Please upload a chest X-ray image before submitting.")
        return redirect(url_for("index"))

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        flash("No file selected. Please choose a PNG, JPG, or JPEG image.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Please upload a PNG, JPG, or JPEG image.")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded_file.filename)
    file_path = UPLOAD_FOLDER / filename
    uploaded_file.save(file_path)

    try:
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        result_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions)) * 100
        prediction = CLASS_NAMES.get(result_index, "UNKNOWN")
    finally:
        if file_path.exists():
            os.remove(file_path)

    return redirect(url_for("result", prediction=prediction, confidence=round(confidence, 2)))

@app.route("/result")
def result():
    return render_template(
        "result.html",
        prediction=request.args.get("prediction", "UNKNOWN"),
        confidence=request.args.get("confidence", "N/A")
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/doctors")
def doctors():
    return render_template("doctors.html", doctors=DOCTORS)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        issue = request.form.get("prediction_issue", "No")
        message = request.form.get("message", "").strip()

        if not message:
            flash("Please describe the issue or feedback before submitting.")
            return redirect(url_for("feedback"))

        file_exists = FEEDBACK_FILE.exists()
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["name", "email", "prediction_issue", "message"])
            writer.writerow([name, email, issue, message])

        flash("Thank you for your feedback. Your response has been recorded.")
        return redirect(url_for("feedback"))

    return render_template("feedback.html")

if __name__ == "__main__":
    app.run(debug=True)
