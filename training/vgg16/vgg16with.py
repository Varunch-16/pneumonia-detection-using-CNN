from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\train'
test_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\test'
val_dir = r'D:\pneumonia_project\efficientNetB0\chest_xray\val'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size
batch_size = 32

# Create generators for training, validation, and testing datasets
train_set = train_datagen.flow_from_directory(train_dir,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              class_mode='binary')

validation_set = test_datagen.flow_from_directory(val_dir,
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='binary')

# Define the VGG16 base model
base_model1 = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="max", classes=2)

# Freeze the layers
for layer in base_model1.layers:
    layer.trainable = False

# Build the model
model = Sequential()
model.add(base_model1)
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model with Adamax optimizer and binary_crossentropy loss
model.compile(optimizer=Adamax(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Print the model summary
model.summary()

# Train the model
epochs = 10
history = model.fit_generator(train_set, epochs=epochs, validation_data=validation_set, steps_per_epoch=len(train_set), validation_steps=len(validation_set))

# Plot training and validation accuracy and loss
plt.figure(figsize=(15, 5))

# Plot Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
train_score = model.evaluate(train_set, verbose=1)
valid_score = model.evaluate(validation_set, verbose=1)
test_score = model.evaluate(test_set, verbose=1)

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
preds = model.predict(test_set)
y_pred = np.round(preds)

cm = confusion_matrix(test_set.classes, y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Pneumonia'], rotation=45)
plt.yticks(tick_marks, ['Normal', 'Pneumonia'])

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.show()

# Print classification report
print(classification_report(test_set.classes, y_pred, target_names=['Normal', 'Pneumonia']))

# Save the model
model.save("pneumonia_detection_model.h5")
print("Model saved as pneumonia_detection_model.h5")
