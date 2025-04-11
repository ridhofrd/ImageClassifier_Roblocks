import os
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

def extract_color_histogram(image, bins=16):
    image = cv2.resize(image, (100, 100))
    chans = cv2.split(image)
    features = []

    for chan in chans: #[RGB]
        hist = cv2.calcHist([chan], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features)

def load_images_from_folder(folder):
    features = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = cv2.imread(path)
        if image is not None:
            features.append(extract_color_histogram(image))
    return features

def train_and_export_model():
    class1 = load_images_from_folder('uploads/class1')
    class2 = load_images_from_folder('uploads/class2')

    X = np.array(class1 + class2)
    y = np.array([0] * len(class1) + [1] * len(class2))
    y_cat = to_categorical(y, num_classes=2)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(48,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y_cat, epochs=100, batch_size=8, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs('model', exist_ok=True)
    with open('model/color_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
