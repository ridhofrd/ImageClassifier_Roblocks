from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_mysqldb import MySQL
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import json
from flask import session
# import yaml # No longer needed for TensorFlow training
import logging # For better logging
import shutil
import re
import os
os.environ['TF_LOGGING_VERBOSITY'] = 'ERROR'  # Suppress absl warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # Disable oneDNN messages

import tensorflow as tf
from tensorflow.lite.python import metadata as _metadata
from tensorflow.lite.tools import flatbuffer_utils
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)
app.secret_key = 'RoblockSecretKey' 


# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


# MySQL Configuration
app.config['MYSQL_HOST'] = os.environ.get('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.environ.get('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.environ.get('MYSQL_DB', 'roblock_module')
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Upload and Model Folders
UPLOAD_FOLDER = 'uploads' # Using lowercase
MODEL_PATH = 'model/model.tflite' # Final TFLite model path
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)


# Helper function for database queries
def execute_query(query, args=None, fetch_one=False, commit=False):
    cur = mysql.connection.cursor()
    cur.execute(query, args)
    if commit:
        mysql.connection.commit()
    result = cur.fetchone() if fetch_one else cur.fetchall()
    cur.close()
    return result

# --- API Image Classifier with TensorFlow ---
@app.route('/train', methods=['POST'])
def train():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    if os.path.exists(session_folder):
        app.logger.warning(f"Session folder already exists: {session_folder}. Overwriting.")
        shutil.rmtree(session_folder)
    
    os.makedirs(session_folder)
    app.logger.info(f"Session folder created: {session_folder}")
    app.logger.info(f"Query params: {request.args}")
    app.logger.info(f"Form data: {request.form}")
    app.logger.info(f"Files received: {len(request.files.getlist('images'))}")

    try:
        # Get parameters and class labels
        class_labels_raw = request.form.getlist('class_label')
        app.logger.info(f"Raw class labels: {class_labels_raw}")
        class_labels_from_form = []

        for label in class_labels_raw:
            if not label:
                continue
            try:
                parsed = json.loads(label)
                if isinstance(parsed, list):
                    class_labels_from_form.extend(parsed)
                elif isinstance(parsed, str):
                    class_labels_from_form.append(parsed)
                else:
                    app.logger.error(f"Invalid class_label format for {label}: expected JSON list or string")
                    return jsonify({"error": f"Invalid class_label format for {label}, expected JSON list or string"}), 400
            except json.JSONDecodeError:
                class_labels_from_form.append(label.strip('"'))
                app.logger.info(f"Treating class_label as plain string: {label}")

        class_labels_from_form = sorted(list(set(class_labels_from_form)))
        if not class_labels_from_form:
            app.logger.error("No valid class labels provided")
            return jsonify({"error": "No valid class labels provided"}), 400
        app.logger.info(f"Intended class labels from form: {class_labels_from_form}")

        train_base_dir = os.path.join(session_folder, 'train')
        val_base_dir = os.path.join(session_folder, 'val')
        os.makedirs(train_base_dir, exist_ok=True)
        os.makedirs(val_base_dir, exist_ok=True)

        for label in class_labels_from_form:
            os.makedirs(os.path.join(train_base_dir, label), exist_ok=True)
            os.makedirs(os.path.join(val_base_dir, label), exist_ok=True)

        app.logger.info(f"Created train_base_dir: {train_base_dir}, val_base_dir: {val_base_dir}")
        epochs = int(request.args.get('epochs', 10))
        batch_size = int(request.args.get('batch_size', 32))
        learning_rate = float(request.args.get('learning_rate', 0.001))
        app.logger.info(f"Parameters - epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")

        image_paths_original_location = []
        image_labels_original = []

        files = request.files.getlist('images')
        if not files or all(not f.filename for f in files):
            app.logger.error("No images provided or images are empty.")
            return jsonify({"error": "No images provided or images are empty"}), 400

        temp_image_save_dir = os.path.join(session_folder, "all_images_temp")
        os.makedirs(temp_image_save_dir, exist_ok=True)

        for idx, file in enumerate(files):
            if not file.filename:
                app.logger.warning(f"Image at index {idx} has no filename, skipping.")
                continue

            filename = secure_filename(file.filename)
            app.logger.info(f"Processing file: {filename}")
            class_label_for_file = ""
            try:
                class_label_for_file = filename.split('_img_')[0]
                app.logger.info(f"Extracted class label: {class_label_for_file}")
                if class_label_for_file not in class_labels_from_form:
                    app.logger.error(f"Class label '{class_label_for_file}' from filename '{filename}' not in defined class_labels: {class_labels_from_form}")
                    return jsonify({"error": f"Class label '{class_label_for_file}' from filename '{filename}' not in defined class_labels: {class_labels_from_form}"}), 400
            except IndexError:
                if len(class_labels_from_form) == len(files):
                    class_label_for_file = class_labels_from_form[idx]
                    app.logger.info(f"Assigned class label from form: {class_label_for_file}")
                else:
                    if len(class_labels_from_form) == 1:
                        class_label_for_file = class_labels_from_form[0]
                        app.logger.info(f"Assigned single class label: {class_label_for_file}")
                    else:
                        app.logger.error(f"Filename '{filename}' does not follow 'class_img_name' pattern and 'class_label' form field count mismatch for fallback.")
                        return jsonify({"error": f"Cannot determine label for '{filename}'. Filename pattern mismatch or ambiguous form labels."}), 400

            if not class_label_for_file:
                app.logger.error(f"Class label for file '{filename}' ended up empty after processing.")
                return jsonify({"error": f"Could not determine class label for file '{filename}'."}), 400

            temp_file_path = os.path.join(temp_image_save_dir, filename)
            file.save(temp_file_path)
            image_paths_original_location.append(temp_file_path)
            image_labels_original.append(class_label_for_file)

        if not image_paths_original_location:
            app.logger.error("No valid images were processed after parsing labels.")
            return jsonify({"error": "No valid images were processed"}), 400

        min_samples_per_class = min([image_labels_original.count(cls) for cls in set(image_labels_original)]) if image_labels_original else 0
        can_stratify = min_samples_per_class >= 2 if len(set(image_labels_original)) > 1 else False

        if can_stratify:
            train_files, val_files, _, _ = train_test_split(
                image_paths_original_location, image_labels_original,
                test_size=0.2, stratify=image_labels_original, random_state=42
            )
        else:
            app.logger.warning("Not enough samples for stratification. Splitting without stratification.")
            train_files, val_files = train_test_split(
                image_paths_original_location, test_size=0.2, random_state=42, shuffle=True
            )

        for file_path in train_files:
            original_index = image_paths_original_location.index(file_path)
            label = image_labels_original[original_index]
            dest_path = os.path.join(train_base_dir, label, os.path.basename(file_path))
            os.rename(file_path, dest_path)
        
        for file_path in val_files:
            original_index = image_paths_original_location.index(file_path)
            label = image_labels_original[original_index]
            dest_path = os.path.join(val_base_dir, label, os.path.basename(file_path))
            os.rename(file_path, dest_path)
        
        if os.path.exists(temp_image_save_dir):
            shutil.rmtree(temp_image_save_dir)

        app.logger.info(f"Train directory: {train_base_dir}, Val directory: {val_base_dir}")

        try:
            train_dataset = tf.keras.utils.image_dataset_from_directory(
                train_base_dir,
                labels='inferred',
                label_mode='categorical',
                image_size=IMAGE_SIZE,
                interpolation='nearest',
                batch_size=batch_size,
                shuffle=True
            )
            val_dataset = tf.keras.utils.image_dataset_from_directory(
                val_base_dir,
                labels='inferred',
                label_mode='categorical',
                image_size=IMAGE_SIZE,
                interpolation='nearest',
                batch_size=batch_size,
                shuffle=False
            )
        except Exception as e:
            app.logger.error(f"Error creating TensorFlow image datasets: {e}")
            return jsonify({"error": f"Failed to load images for training. Ensure each class has images. Details: {str(e)}"}), 500

        actual_class_names = train_dataset.class_names
        num_classes = len(actual_class_names)
        if num_classes == 0:
            app.logger.error("TensorFlow inferred 0 classes. Check directory structure.")
            return jsonify({"error": "No classes found in training data."}), 400
        app.logger.info(f"TensorFlow inferred class names: {actual_class_names}, Num classes: {num_classes}")

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        
        def preprocess_dataset(image, label):
            image = normalization_layer(image)
            return image, label

        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(preprocess_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(preprocess_dataset, num_parallel_calls=tf.data.AUTOTUNE)

        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        x = inputs
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        app.logger.info("Model compiled.")
        model.summary(print_fn=app.logger.info)

        app.logger.info(f"Starting training for {epochs} epochs...")
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
        )

        metrics_history = {
            "accuracy": history.history['accuracy'],
            "val_accuracy": history.history.get('val_accuracy', []),
            "loss": history.history['loss'],
            "val_loss": history.history.get('val_loss', [])
        }
        final_val_accuracy = metrics_history['val_accuracy'][-1] if metrics_history['val_accuracy'] else 0.0

        keras_model_dir = os.path.join(session_folder, 'keras_saved_model')
        model.export(keras_model_dir)
        app.logger.info(f"Keras model exported to: {keras_model_dir}")

        converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.target_spec.supported_types = [tf.float32]
        try:
            tflite_model_content = converter.convert()
        except Exception as e:
            app.logger.error(f"TFLite conversion failed: {str(e)}")
            return jsonify({"error": f"TFLite conversion failed: {str(e)}"}), 500

        temp_tflite_path = os.path.join(session_folder, 'model.tflite')
        with open(temp_tflite_path, 'wb') as f:
            f.write(tflite_model_content)

        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata_fb
        from tflite_support.metadata_writers import image_classifier
        from tflite_support.metadata_writers import writer_utils

        ImageClassifierWriter = image_classifier.MetadataWriter

        _INPUT_NORM_MEAN = 0.0
        _INPUT_NORM_STD = 255.0

        input_meta = _metadata_fb.TensorMetadata()
        input_meta.name = "image"
        input_meta.description = "Input image to be classified."
        input_meta.content = _metadata_fb.Content()
        input_meta.content.contentProperties = _metadata_fb.ImageProperties()
        input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

        norm_options = _metadata_fb.ProcessUnit()
        norm_options.options = _metadata_fb.NormalizationOptions()
        norm_options.options.mean = [_INPUT_NORM_MEAN]
        norm_options.options.std = [_INPUT_NORM_STD]
        input_meta.processUnits = [norm_options]

        output_meta = _metadata_fb.TensorMetadata()
        output_meta.name = "probability"
        output_meta.description = "Probabilities of the classified categories."

        writer = ImageClassifierWriter.create_from_metadata(
            tflite_model_content,
            input_meta,
            output_meta,
            label_files=[writer_utils.LabelFile(file_path=_LABEL_FILE)]
        )

        writer_utils.save_file(writer.populate(), temp_tflite_path)
        app.logger.info(f"TFLite model with CORRECTED metadata saved to: {temp_tflite_path}")

        MODEL_PATH = temp_tflite_path

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "model_path": MODEL_PATH,
            "class_names_inferred": actual_class_names,
            "final_val_accuracy": final_val_accuracy,
            "metrics_history": metrics_history,
            "message": "Model trained and saved successfully.",
            "class_names": actual_class_names,
            "session_folder": session_folder,
            "num_classes": num_classes,
            "epochs": epochs
        }), 200

    except Exception as e:
        app.logger.exception("Unexpected error during training")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    session_id = request.args.get('session_id')
    if not session_id or not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        app.logger.error(f"Invalid or missing session_id: {session_id}")
        return jsonify({"error": "Invalid or missing session_id. Use alphanumeric characters, underscores, or hyphens only."}), 400

    session_folder = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    model_path = os.path.join(session_folder, 'model.tflite')
    
    if not os.path.exists(model_path):
        app.logger.error(f"Model not found at: {model_path}")
        return jsonify({"error": f"Model not found at {model_path}"}), 404
    
    app.logger.info(f"Serving model file: {model_path}")
    try:
        return send_file(model_path, as_attachment=True, download_name='model.tflite')
    except Exception as e:
        app.logger.error(f"Failed to send model file {model_path}: {str(e)}")
        return jsonify({"error": f"Failed to send model file: {str(e)}"}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return "pong!"

@app.route('/test', methods=['GET'])
def test():
    return "Server is running!"

if __name__ == '__main__':
    import os
    extra_dirs = [os.getcwd()] 
    extra_files = []
    app.run(host='0.0.0.0', port=5000, extra_files=extra_files, debug=True, use_reloader= False)
