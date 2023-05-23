#!/usr/bin/env python3
# import packages
import os
import cv2
#scikit-learn
from sklearn.model_selection import train_test_split
# VGG16 mode and preproccessing
from tensorflow.keras.applications.vgg16 import (preprocess_input, decode_predictions, VGG16)
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy.linalg import norm
#plotting
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Help functions

def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()



#load and preprocces data
def load_data():
    # Set the path to your dataset
    dataset_path = os.path.join("in", "dataset")

    # Define the labels
    weather_conditions = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

    # Prepare the dataset
    images = []
    labels = []
    for condition_idx, condition in enumerate(weather_conditions):
        condition_path = os.path.join(dataset_path, condition)
        for image_file in os.listdir(condition_path):
            image_path = os.path.join(condition_path, image_file)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = cv2.resize(image, (224, 224))  # Resize to a consistent shape
                images.append(image)
                labels.append(condition_idx)
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(f"Error message: {str(e)}")
    # Convert lists to arrays
    images = np.array(images)
    labels = np.array(labels)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Preprocess the images as required by VGG16
    #X_train = VGG16.preprocess_input(X_train)
    #X_test = VGG16.preprocess_input(X_test)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_test, X_train, y_train, y_test, weather_conditions


def run_model(X_test, X_train, y_train, y_test, weather_conditions):
    # Load the pre-trained VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Extract features from VGG16
    features_train = np.array([extract_features(img_path, vgg_model) for img_path in X_train])
    features_test = np.array([extract_features(img_path, vgg_model) for img_path in X_test])

    # Build a classifier on top of the extracted features
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=features_train.shape[1:]))
    model.add(Dense(len(weather_conditions), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    H = model.fit(features_train, y_train, epochs=10, batch_size=32, validation_data=(features_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(features_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    return H
    

def plot_history(y_test, y_pred, weather_conditions, H):
    report = classification_report(y_test, y_pred, target_names=weather_conditions)
    plot_history(H, 10)


def main():
    X_test, X_train, y_train, y_test, weather_conditions = load_data()
    H = run_model(X_train, X_test, y_train, y_test, weather_conditions)
    plot_history(y_test, y_pred, weather_conditions, H)

if __name__=="__main__":
    main()
