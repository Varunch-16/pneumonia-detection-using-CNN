import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = load_model('without\pneumonia_detection_model_efficientnetb0_updated.h5')
img_size = (224, 224)
batch_size = 32
# Directories for test data
test_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\test'

# Create a test data generator
test_gen = ImageDataGenerator(preprocessing_function=lambda img: img).flow_from_directory(test_dir, target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

# Evaluate the model on the test data
test_accuracy = model.evaluate(test_gen)[1]

# Print test accuracy
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
