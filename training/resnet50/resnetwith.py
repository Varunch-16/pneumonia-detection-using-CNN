import os
import time
import shutil
import pathlib
import itertools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
import warnings
warnings.filterwarnings("ignore")

print('Modules loaded')

# Specify your file paths
train_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\train'
test_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\test'
val_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\val'

# Create filepaths and labels
filepaths = []
labels = []

# Update data_dir with the new paths
for data_dir in [train_dir, test_dir, val_dir]:
    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

# Create DataFrame
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)

# Split data into train, validation, and test sets
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

# Set up data generators with the specified directories
batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator(preprocessing_function=lambda img: img)
ts_gen = ImageDataGenerator(preprocessing_function=lambda img: img)

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size, directory=train_dir)

valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size, directory=val_dir)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size, directory=test_dir)

# Build the model
class_count = len(list(train_gen.class_indices.keys()))
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_shape)
model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(class_count, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
epochs = 10
history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, validation_steps=None, shuffle=False)

# Plot training and validation accuracy and loss
plt.figure(figsize=(15, 5))

# Plot Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

# Print accuracy and loss in percentage
train_accuracy_percentage = train_score[1] * 100
valid_accuracy_percentage = valid_score[1] * 100
test_accuracy_percentage = test_score[1] * 100

train_loss_percentage = train_score[0] * 100
valid_loss_percentage = valid_score[0] * 100
test_loss_percentage = test_score[0] * 100

print("Train Accuracy: {:.2f}%".format(train_accuracy_percentage))
print("Validation Accuracy: {:.2f}%".format(valid_accuracy_percentage))
print("Test Accuracy: {:.2f}%".format(test_accuracy_percentage))

print("Train Loss: {:.2f}%".format(train_loss_percentage))
print("Validation Loss: {:.2f}%".format(valid_loss_percentage))
print("Test Loss: {:.2f}%".format(test_loss_percentage))

# Visualize confusion matrix
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()

# Print classification report
print(classification_report(test_gen.classes, y_pred, target_names=classes))

# Save the model
model.save('vgg16with.h5')
print('Model saved as pneumonia_detection_model.h5')
