import os
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from keras.utils import to_categorical

def extract_dominant_color(image):
    image = cv2.resize(image, (50, 50))
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color / 255.0  # Normalized RGB

def load_images_from_folder(folder):
    features = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = cv2.imread(path)
        if image is not None:
            rgb = extract_dominant_color(image)
            features.append(rgb)
    return features

def train_and_export_model():
    class1_images = load_images_from_folder('uploads/class1')
    class2_images = load_images_from_folder('uploads/class2')

    X = np.array(class1_images + class2_images)
    y = np.array([0] * len(class1_images) + [1] * len(class2_images))

    y_cat = to_categorical(y, num_classes=2)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=50, verbose=0)

    # Save model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs('model', exist_ok=True)
    with open('model/color_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
