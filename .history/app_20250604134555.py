from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_mysqldb import MySQL
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import json
# import yaml # No longer needed for TensorFlow training
import logging # For better logging

import tensorflow as tf
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

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

global session_id 
global session_folder
session_id = None
session_folder = None

# --- API Image Classifier with TensorFlow ---
@app.route('/train', methods=['POST'])
def train():
    session_id = str(uuid.uuid4())
    session_folder = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    os.makedirs(session_folder, exist_ok=True)
    app.logger.info(f"Session folder created: {session_folder}")
    app.logger.info(f"Query params: {request.args}")
    app.logger.info(f"Form data: {request.form}")
    app.logger.info(f"Files received: {len(request.files.getlist('images'))}")

    try:
        # Get parameters and class labels
        class_labels_raw = request.form.getlist('class_label')  # Get all class_label entries
        app.logger.info(f"Raw class labels: {class_labels_raw}")
        class_labels_from_form = []

        for label in class_labels_raw:
            if not label:
                continue
            try:
                # Try parsing as JSON (handles quoted strings or JSON lists)
                parsed = json.loads(label)
                if isinstance(parsed, list):
                    class_labels_from_form.extend(parsed)  # Add all items from JSON list
                elif isinstance(parsed, str):
                    class_labels_from_form.append(parsed)  # Add single string
                else:
                    app.logger.error(f"Invalid class_label format for {label}: expected JSON list or string")
                    return jsonify({"error": f"Invalid class_label format for {label}, expected JSON list or string"}), 400
            except json.JSONDecodeError:
                # Treat as plain string if JSON parsing fails (e.g., "Class_1" or Class_1)
                class_labels_from_form.append(label.strip('"'))  # Remove quotes if present
                app.logger.info(f"Treating class_label as plain string: {label}")

        # Remove duplicates, sort, and validate
        class_labels_from_form = sorted(list(set(class_labels_from_form)))
        if not class_labels_from_form:
            app.logger.error("No valid class labels provided")
            return jsonify({"error": "No valid class labels provided"}), 400
        app.logger.info(f"Intended class labels from form: {class_labels_from_form}")

        train_base_dir = os.path.join(session_folder, 'train')
        val_base_dir = os.path.join(session_folder, 'val')
        os.makedirs(train_base_dir, exist_ok=True)
        os.makedirs(val_base_dir, exist_ok=True)

        # Create class-specific subdirectories under train and val
        for label in class_labels_from_form:
            os.makedirs(os.path.join(train_base_dir, label), exist_ok=True)
            os.makedirs(os.path.join(val_base_dir, label), exist_ok=True)

        app.logger.info(f"Created train_base_dir: {train_base_dir}, val_base_dir: {val_base_dir}")
        epochs = int(request.args.get('epochs', 10))
        batch_size = int(request.args.get('batch_size', 32))
        learning_rate = float(request.args.get('learning_rate', 0.001))
        app.logger.info(f"Parameters - epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")

        if not class_labels_from_form:
            app.logger.error("No valid class labels after processing")
            return jsonify({"error": "No valid class labels provided"}), 400
        app.logger.info(f"Intended class labels from form: {class_labels_from_form}")

        # Prepare image file paths and labels
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

        # Handle single image case (temporary for testing)
        if len(image_paths_original_location) < 2:
            app.logger.warning("Only one image provided; skipping train/validation split for testing.")
            # For testing, proceed with single image; adjust for production
            # return jsonify({"error": "Not enough images to split for training and validation. Need at least 2."}), 400
        
        min_samples_per_class = min([image_labels_original.count(cls) for cls in set(image_labels_original)]) if image_labels_original else 0
        can_stratify = min_samples_per_class >= 2 if len(set(image_labels_original)) > 1 else False

        if can_stratify:
            train_files, val_files, _, _ = train_test_split(
                image_paths_original_location, image_labels_original, 
                test_size=0.2, stratify=image_labels_original, random_state=42
            )
        else:
            if not can_stratify and len(set(image_labels_original)) > 1:
                 app.logger.warning("Not enough samples in at least one class for stratification. Splitting without stratification.")
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
            import shutil
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
            app.logger.error(f"Error creating TensorFlow image datasets: {e}. Check if all class folders under train/val have images.")
            return jsonify({"error": f"Failed to load images for training. Ensure each class has images in train and val sets. Details: {str(e)}"}), 500

        actual_class_names = train_dataset.class_names
        num_classes = len(actual_class_names)
        if num_classes == 0:
            app.logger.error("TensorFlow inferred 0 classes. Train/Val directories might be empty or structured incorrectly.")
            return jsonify({"error": "No classes found in the training data. Please check directory structure and image uploads."}), 400
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
        app.logger.info("TensorFlow model compiled.")
        model.summary(print_fn=app.logger.info)

        app.logger.info(f"Starting TensorFlow model training for {epochs} epochs...")
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
        )
        app.logger.info("TensorFlow model training completed.")

        keras_model_dir = os.path.join(session_folder, 'keras_saved_model')
        model.export(keras_model_dir) 
        app.logger.info(f"Keras model exported to SavedModel format at: {keras_model_dir}")


        converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT] 
        tflite_model_content = converter.convert()

        temp_tflite_path = os.path.join(session_folder, 'model.tflite')
        with open(temp_tflite_path, 'wb') as f:
            f.write(tflite_model_content)
        app.logger.info(f"TFLite model converted and saved to: {temp_tflite_path}")

        # Ensure target directory exists and remove old model if it exists
        final_model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(final_model_dir, exist_ok=True) 
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            app.logger.info(f"Removed existing model at: {MODEL_PATH}")
        
        os.rename(temp_tflite_path, MODEL_PATH)
        app.logger.info(f"TFLite model moved to final path: {MODEL_PATH}")
        db_model_path = MODEL_PATH 

        #nanti kalau udah ada server di aktifin lagi

        # query = """
        #     INSERT INTO Training_sessions
        #     (session_id, class_labels, model_path, created_at)
        #     VALUES (%s, %s, %s, %s)
        # """
        # execute_query(query, (session_id, ','.join(actual_class_names), db_model_path, int(datetime.now().timestamp() * 1000)), commit=True)

        training_metrics = {
            "epochs": list(range(1, len(history.history['loss']) + 1)),
            "loss": history.history.get('loss', []),
            "accuracy": history.history.get('accuracy', []),
            "val_loss": history.history.get('val_loss', []),
            "val_accuracy": history.history.get('val_accuracy', [])
        }
        final_val_accuracy = history.history['val_accuracy'][-1] if history.history.get('val_accuracy') else 0.0

        app.logger.info("Training successful.")
        return jsonify({
            "status": "success", 
            "session_id": session_id, 
            "model_path": db_model_path, 
            "class_names_inferred": actual_class_names,
            "final_val_accuracy": final_val_accuracy,
            "metrics_history": training_metrics
        }), 200

    except Exception as e:
        app.logger.error(f"Training Error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/download_model', methods=['GET'])
def download_model():   
    # Always serve the same model file (e.g., 'model.tflite' in a known directory)
    model_to_download_path = os.path.abspath()
    if os.path.exists(model_to_download_path):
        return send_file(model_to_download_path, as_attachment=True)
    return jsonify({"error": "Model not found"}), 404

# --- API Module and Quiz for Android App ---
@app.route('/api/modules', methods=['GET'])
def api_get_all_modules():
    query = "SELECT * FROM Module_table"
    modules = execute_query(query)
    return jsonify(modules)

@app.route('/api/modules/<module_id>', methods=['GET'])
def api_get_module(module_id):
    query = "SELECT * FROM Module_table WHERE id = %s"
    module = execute_query(query, (module_id,), fetch_one=True)
    if module:
        return jsonify(module)
    return jsonify({"error": "Module not found"}), 404

@app.route('/api/modules/<module_id>/questions', methods=['GET'])
def api_get_module_questions(module_id):
    query = "SELECT * FROM Question_table WHERE module_id = %s"
    questions = execute_query(query, (module_id,))
    return jsonify(questions)

# --- Web Interface Module and Quiz ---
@app.route('/')
@app.route('/dashboard')
def dashboard():
    try:
        query = "SELECT COUNT(*) as total_modules FROM Module_table"
        result = execute_query(query, fetch_one=True)
        total_modules = result['total_modules'] if result else 0
    except Exception as e:
        app.logger.error(f"Error fetching dashboard data: {e}")
        total_modules = "Error" 
    return render_template('dashboard.html', total_modules=total_modules)

@app.route('/modules')
def manage_modules():
    try:
        query = "SELECT * FROM Module_table"
        modules = execute_query(query)
    except Exception as e:
        app.logger.error(f"Error fetching modules: {e}")
        modules = [] 
    return render_template('modules.html', modules=modules)

@app.route('/modules/create', methods=['GET', 'POST'])
def create_module():
    if request.method == 'POST':
        try:
            data = request.form
            query = """
                INSERT INTO Module_table
                (id, title, description, created_at, updated_at, link_video)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            timestamp = int(datetime.now().timestamp() * 1000)
            args = (
                data.get('id'), 
                data.get('title'), 
                data.get('description'),
                timestamp, 
                timestamp, 
                data.get('link_video')
            )
            execute_query(query, args, commit=True)
            return redirect(url_for('manage_modules'))
        except Exception as e:
            app.logger.error(f"Error creating module: {e}")
            return "Error creating module", 500
    return render_template('create_module.html')

@app.route('/modules/<module_id>/edit', methods=['GET', 'POST'])
def edit_module(module_id):
    if request.method == 'POST':
        try:
            data = request.form
            query = """
                UPDATE Module_table
                SET title = %s, description = %s, updated_at = %s, link_video = %s
                WHERE id = %s
            """
            timestamp = int(datetime.now().timestamp() * 1000)
            args = (
                data.get('title'), 
                data.get('description'),
                timestamp, 
                data.get('link_video'), 
                module_id
            )
            execute_query(query, args, commit=True)
            return redirect(url_for('manage_modules'))
        except Exception as e:
            app.logger.error(f"Error editing module {module_id}: {e}")
            return f"Error editing module {module_id}", 500
            
    try:
        query = "SELECT * FROM Module_table WHERE id = %s"
        module = execute_query(query, (module_id,), fetch_one=True)
        if not module:
            return "Module not found", 404
        return render_template('edit_module.html', module=module)
    except Exception as e:
        app.logger.error(f"Error fetching module {module_id} for edit: {e}")
        return f"Error fetching module {module_id}", 500


@app.route('/modules/<module_id>/delete', methods=['POST'])
def delete_module(module_id):
    try:
        query = "DELETE FROM Module_table WHERE id = %s"
        execute_query(query, (module_id,), commit=True)
        return redirect(url_for('manage_modules'))
    except Exception as e:
        app.logger.error(f"Error deleting module {module_id}: {e}")
        return f"Error deleting module {module_id}", 500

@app.route('/modules/<module_id>/questions')
def manage_questions(module_id):
    try:
        module_query = "SELECT * FROM Module_table WHERE id = %s"
        module = execute_query(module_query, (module_id,), fetch_one=True)
        if not module:
            return "Module not found", 404

        questions_query = "SELECT * FROM Question_table WHERE module_id = %s"
        questions = execute_query(questions_query, (module_id,))
        return render_template('questions.html', module=module, questions=questions)
    except Exception as e:
        app.logger.error(f"Error managing questions for module {module_id}: {e}")
        return f"Error managing questions for module {module_id}", 500


@app.route('/modules/<module_id>/questions/create', methods=['GET', 'POST'])
def create_question(module_id):
    if request.method == 'POST':
        try:
            data = request.form
            query = """
                INSERT INTO Question_table
                (id, module_id, question_text, option_a, option_b, option_c, option_d, correct_answer, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            timestamp = int(datetime.now().timestamp() * 1000)
            args = (
                data.get('id'), module_id, data.get('question_text'),
                data.get('option_a'), data.get('option_b'), data.get('option_c'), data.get('option_d'),
                data.get('correct_answer'), timestamp, timestamp
            )
            execute_query(query, args, commit=True)
            return redirect(url_for('manage_questions', module_id=module_id))
        except Exception as e:
            app.logger.error(f"Error creating question for module {module_id}: {e}")
            return f"Error creating question for module {module_id}", 500
    return render_template('create_question.html', module_id=module_id)

@app.route('/questions/<question_id>/edit', methods=['GET', 'POST'])
def edit_question(question_id):
    try:
        module_id_query = "SELECT module_id FROM Question_table WHERE id = %s"
        question_data = execute_query(module_id_query, (question_id,), fetch_one=True)
        if not question_data:
            return "Question not found to determine module", 404
        redirect_module_id = question_data['module_id']

        if request.method == 'POST':
            data = request.form
            query = """
                UPDATE Question_table
                SET question_text = %s, option_a = %s, option_b = %s,
                    option_c = %s, option_d = %s, correct_answer = %s, updated_at = %s
                WHERE id = %s
            """
            timestamp = int(datetime.now().timestamp() * 1000)
            args = (
                data.get('question_text'), data.get('option_a'), data.get('option_b'),
                data.get('option_c'), data.get('option_d'), data.get('correct_answer'),
                timestamp, question_id
            )
            execute_query(query, args, commit=True)
            return redirect(url_for('manage_questions', module_id=redirect_module_id))

        query_select = "SELECT * FROM Question_table WHERE id = %s" 
        question = execute_query(query_select, (question_id,), fetch_one=True)
        if not question:
            return "Question not found", 404
        return render_template('edit_question.html', question=question)
    except Exception as e:
        app.logger.error(f"Error editing question {question_id}: {e}")
        return f"Error editing question {question_id}", 500


@app.route('/questions/<question_id>/delete', methods=['POST'])
def delete_question(question_id):
    try:
        module_query = "SELECT module_id FROM Question_table WHERE id = %s"
        question = execute_query(module_query, (question_id,), fetch_one=True)
        if not question:
            app.logger.warning(f"Attempted to delete non-existent question or question without module_id: {question_id}")
            return redirect(url_for('manage_modules')) 

        query_delete = "DELETE FROM Question_table WHERE id = %s" 
        execute_query(query_delete, (question_id,), commit=True)
        return redirect(url_for('manage_questions', module_id=question['module_id']))
    except Exception as e:
        app.logger.error(f"Error deleting question {question_id}: {e}")
        return f"Error deleting question {question_id}", 500


# --- Additional Endpoints ---
@app.route('/upload', methods=['POST'])
def upload_image(): # This is a generic image uploader, not used by /train
    label = request.form.get('label')
    image = request.files.get('image')

    if not label or not image or not image.filename:
        return jsonify({'error': 'Label and image file are required.'}), 400

    label_folder = os.path.join(UPLOAD_FOLDER, secure_filename(label)) 
    os.makedirs(label_folder, exist_ok=True)

    filename = secure_filename(image.filename) # Corrected: removed .keras concatenation
    image_path = os.path.join(label_folder, filename)
    image.save(image_path)

    return jsonify({'message': f'Image saved to {image_path}'}), 200

@app.route('/ping', methods=['GET'])
def ping():
    return "pong!"

@app.route('/test', methods=['GET'])
def test():
    return "Server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
