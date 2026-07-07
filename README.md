# Pneumonia Detection in Chest X-Ray Images Using EfficientNetB0

This project is a deep learning-based web application designed to detect pneumonia from chest X-ray images. The system uses a trained EfficientNetB0 model to classify an uploaded X-ray image as either Normal or Pneumonia.

The project was developed as part of an academic research work and is associated with the published paper titled Pneumonia Detection in Chest X-Ray Images Using EfficientNetB0, published in the International Journal of Research and Analytical Reviews, June 2024.

## Project Overview

Pneumonia is a serious lung infection that is commonly diagnosed using chest X-ray imaging. Manual interpretation of X-ray images requires trained medical professionals and may lead to delays, especially in areas with limited healthcare access.

The main goal of this project is to support faster preliminary screening of pneumonia using deep learning. Several transfer learning models were trained and evaluated on chest X-ray images. After comparing their performance, EfficientNetB0 was selected for deployment in the final web application.

The trained EfficientNetB0 model was saved as an H5 file and integrated with a Flask backend. The web interface allows users to upload a chest X-ray image and view the prediction result directly through the browser.

## Key Features

- Chest X-ray image upload through a web interface
- Pneumonia classification using a trained EfficientNetB0 model
- Flask-based backend for model loading and prediction
- Simple and responsive HTML/CSS user interface
- Published research paper included in the repository
- Training files included for reference
- Dataset source provided through Kaggle instead of uploading the full dataset

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Flask
- NumPy
- Pillow
- HTML
- CSS

## Model Development

The project followed a transfer learning approach using pre-trained convolutional neural network architectures. Four models were trained and compared for pneumonia detection from chest X-ray images:

- EfficientNetB0
- DenseNet121
- ResNet50
- VGG16

The models were evaluated based on their classification performance. EfficientNetB0 was selected as the final deployment model because it provided strong accuracy while maintaining an efficient architecture suitable for web application integration.

## Model Performance

| Model | Test Accuracy with Filters |
|---|---:|
| EfficientNetB0 | 98.21% |
| DenseNet121 | 98.21% |
| ResNet50 | 90.18% |
| VGG16 | 88.78% |

EfficientNetB0 and DenseNet121 achieved the highest test accuracy. EfficientNetB0 was used in the final Flask application because of its strong validation performance and efficient design.

## Dataset

The dataset used for this project is the Chest X-Ray Images Pneumonia dataset available on Kaggle.

Dataset link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains chest X-ray images organized into two categories:
NORMAL
PNEUMONIA
The full dataset is not included in this repository because it is approximately 1.5 GB in size. Users who want to retrain or test the model can download the dataset directly from Kaggle using the link above.

Project Structure
```text
pneumonia-detection-efficientnetb0/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── new.h5
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   └── images/
│
├── docs/
│   └── IJRAR24B3948.pdf
│
├── training/
│   ├── efficientnetb0/
│   ├── densenet121/
│   ├── resnet50/
│   └── vgg16/
│
├── sample_images/
│
└── screenshots/
How the Application Works

The application follows a simple prediction workflow:

The user uploads a chest X-ray image through the web interface.
The Flask backend receives the uploaded image.
The image is resized and preprocessed to match the model input format.
The trained EfficientNetB0 H5 model performs the prediction.
The result is displayed on the web page as either Normal or Pneumonia.
Installation and Setup
1. Clone the repository
git clone https://github.com/varun22ch/pneumonia-detection-efficientnetb0.git
cd pneumonia-detection-efficientnetb0
2. Create a virtual environment

For Windows:

py -3.11 -m venv venv
3. Activate the virtual environment

For Command Prompt:

venv\Scripts\activate

For PowerShell:

.\venv\Scripts\Activate.ps1
4. Install the required packages
pip install -r requirements.txt
5. Run the Flask application
python app.py
6. Open the application in a browser
http://127.0.0.1:5000
Sample Output

After uploading a chest X-ray image, the application displays one of the following prediction results:

Prediction: Normal

or

Prediction: Pneumonia
Publication

This project is associated with the following published research paper:

Title: Pneumonia Detection in Chest X-Ray Images Using EfficientNetB0
Journal: International Journal of Research and Analytical Reviews
Volume: 11
Issue: 2
Published: June 2024
Paper ID: IJRAR24B3948

Authors:

KVBL Deepthi
P. Nikitha
J. Pragna
CH. Sai Varun

The full paper is included in the docs folder of this repository.

Future Improvements
Add Grad-CAM visualization to show the image regions influencing the prediction
Display prediction confidence score on the result page
Improve the user interface for a better user experience
Deploy the application using a cloud platform
Add support for detecting additional chest-related conditions
Improve model explainability for medical imaging use cases
Important Disclaimer

This project is intended for academic and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Any medical decision should be made by a qualified healthcare professional.

Author

CH. Sai Varun
GitHub: https://github.com/varun22ch

License

This project is intended for academic and educational use.