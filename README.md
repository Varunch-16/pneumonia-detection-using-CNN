# PneumoScan — Pneumonia Detection Web Application

PneumoScan is an academic deep learning web application for preliminary pneumonia screening from chest X-ray images. The project connects an EfficientNetB0-based model with a Flask web application so users can upload an X-ray image, view the prediction result, read general precautions, search for nearby doctors, and submit feedback.

## Important Disclaimer

This project is for academic and educational purposes only. It is not a medical diagnosis tool. Users should consult a qualified healthcare professional for medical advice, diagnosis, or treatment.

## Features

- Chest X-ray image upload
- EfficientNetB0-based prediction workflow
- Result page for Normal or Pneumonia output
- General precaution guidance when pneumonia signs are detected
- Doctor search support using Google Maps search links
- About page written for non-technical users
- Feedback page for reporting prediction or website issues
- Clean Flask-only project structure

## Tech Stack

Python, Flask, TensorFlow/Keras, EfficientNetB0, HTML, CSS, JavaScript, CSV-based feedback storage

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Put your trained model file at:

```text
model/new.h5
```

Then open:

```text
http://127.0.0.1:5000
```

## Published Paper

**Title:** Pneumonia Detection in Chest X-ray Images Using EfficientNetB0  
**Journal:** International Journal of Research and Analytical Reviews  
**Volume/Issue:** Volume 11, Issue 2  
**Publication Date:** June 2024  
**Paper ID:** IJRAR24B3948  

## Author

Sai Varun Chetrypally  
M.S. Computing and Information Systems  
Youngstown State University
