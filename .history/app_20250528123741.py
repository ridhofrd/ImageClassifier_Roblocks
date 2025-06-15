from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_mysqldb import MySQL
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import logging # For better logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO) # You can set this to logging.DEBUG for more detail
app.logger.setLevel(logging.INFO)


# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'roblock_module'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Upload and Model Folders
UPLOAD_FOLDER = 'uploads' # Using lowercase as per our fix
MODEL_PATH = 'model/model.tflite'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper function for database queries
def execute_query(query, args=None, fetch_one=False, commit=False):
    cur = mysql.connection.cursor()
    cur.execute(query, args)
    if commit:
        mysql.connection.commit()
    result = cur.fetchone() if fetch_one else cur.fetchall()
    cur.close()
    return result

# --- API Image Classifier ---
@app.route('/train', methods=['POST'])
def train():
    session_id = str(uuid.uuid4())
    session_folder = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    os.makedirs(session_folder, exist_ok=True)
    app.logger.info(f"Session folder created: {session_folder}")

    try:
        # --- Start: Define class_names, parameters, and process images ---
        class_labels_form = request.form.getlist('class_label')
        # Get epochs, batch_size, learning_rate from form data (for POST) or args (if GET, but POST is better for forms)
        epochs = int(request.form.get('epochs', 10))
        batch_size = int(request.form.get('batch_size', 32))
        learning_rate = float(request.form.get('learning_rate', 0.001))

        if not class_labels_form:
            app.logger.error("No class labels provided in form.")
            return jsonify({"error": "No class labels provided"}), 400

        class_names = sorted(list(set(class_labels_form))) # Define class_names
        app.logger.info(f"Class names defined: {class_names}")

        image_paths_original_location = []
        image_labels_original = []

        files = request.files.getlist('images')
        if not files or all(not f.filename for f in files):
            app.logger.error("No images provided or images are empty.")
            return jsonify({"error": "No images provided or images are empty"}), 400

        temp_image_save_dir = os.path.join(session_folder, "all_images_temp")
        os.makedirs(temp_image_save_dir, exist_ok=True)

        for idx, file in enumerate(files):
            if file.filename == '':
                app.logger.warning(f"Image at index {idx} has no filename, skipping.")
                continue

            filename = secure_filename(file.filename)
            class_label_for_file = ""
            try:
                class_label_for_file = filename.split('_img_')[0]
                if class_label_for_file not in class_names:
                     app.logger.error(f"Class label '{class_label_for_file}' from filename '{filename}' not in defined class_labels: {class_names}")
                     return jsonify({"error": f"Class label '{class_label_for_file}' from filename '{filename}' not in defined class_labels: {class_names}"}), 400
            except IndexError:
                if len(class_labels_form) == len(files):
                    class_label_for_file = class_labels_form[idx]
                    if class_label_for_file not in class_names:
                        app.logger.error(f"Assigned class_label '{class_label_for_file}' for '{filename}' via form order is not in defined class_names: {class_names}")
                        return jsonify({"error": f"Assigned class_label '{class_label_for_file}' for '{filename}' is not a valid defined class."}), 400
                else:
                    app.logger.error(f"Filename '{filename}' does not follow 'class_img_name' pattern and 'class_label' form field count mismatch for fallback.")
                    return jsonify({"error": f"Cannot determine label for '{filename}'. Filename pattern mismatch and form label count incorrect."}), 400
            
            if not class_label_for_file:
                 app.logger.error(f"Class label for file '{filename}' ended up empty.")
                 return jsonify({"error": f"Could not determine class label for file '{filename}'."}), 400

            temp_file_path = os.path.join(temp_image_save_dir, filename)
            file.save(temp_file_path)
            image_paths_original_location.append(temp_file_path)
            image_labels_original.append(class_label_for_file)

        if not image_paths_original_location:
             app.logger.error("No valid images were processed after parsing labels.")
             return jsonify({"error": "No valid images were processed"}), 400
        
        if len(set(image_labels_original)) < len(class_names):
             missing_classes = set(class_names) - set(image_labels_original)
             app.logger.error(f"Declared classes {missing_classes} have no associated images after processing.")
             return jsonify({"error": f"Classes {missing_classes} have no images. Check filenames and labels."}), 400
        
        min_samples_per_class = min([image_labels_original.count(cls) for cls in class_names]) if class_names else 0
        test_size_val = 0.2
        
        if len(image_paths_original_location) <= 1:
             return jsonify({"error": "Not enough images to split. Need at least 2."}), 400

        can_stratify = min_samples_per_class >= 2 

        if can_stratify:
            train_images_paths, val_images_paths, train_labels_list, val_labels_list = train_test_split(
                image_paths_original_location, image_labels_original, test_size=test_size_val, stratify=image_labels_original, random_state=42
            )
        else:
            app.logger.warning("Not enough samples in at least one class for stratification. Splitting without stratification.")
            train_images_paths, val_images_paths, train_labels_list, val_labels_list = train_test_split(
                image_paths_original_location, image_labels_original, test_size=test_size_val, random_state=42, shuffle=True
            )
        # --- End: Image processing and splitting ---

        train_base_dir = os.path.join(session_folder, 'train')
        val_base_dir = os.path.join(session_folder, 'val')
        os.makedirs(train_base_dir, exist_ok=True)
        os.makedirs(val_base_dir, exist_ok=True)
        app.logger.info(f"Train base directory: {train_base_dir}")
        app.logger.info(f"Validation base directory: {val_base_dir}")

        # Create class subdirectories using defined class_names
        for class_name_item in class_names:
            os.makedirs(os.path.join(train_base_dir, class_name_item), exist_ok=True)
            os.makedirs(os.path.join(val_base_dir, class_name_item), exist_ok=True)

        # Move images to structured folders
        for img_path, label in zip(train_images_paths, train_labels_list):
            dest_path = os.path.join(train_base_dir, label, os.path.basename(img_path))
            os.rename(img_path, dest_path)
        for img_path, label in zip(val_images_paths, val_labels_list):
            dest_path = os.path.join(val_base_dir, label, os.path.basename(img_path))
        
        if not os.path.isdir(train_base_dir) or not os.path.isdir(val_base_dir): # Basic check
            app.logger.error(f"CRITICAL: Train or Val base directory does not exist or is not a directory before training.")
            return jsonify({"error": "Train/Val directory failed to be created properly."}), 500

        # data_config = {
        #     'train': os.path.abspath(train_base_dir),
        #     'val': os.path.abspath(val_base_dir),
        #     'nc': len(class_names),
        #     'names': class_names
        # }
        # data_yaml_path = os.path.join(session_folder, 'data.yaml')
        # with open(data_yaml_path, 'w') as f:
        #     yaml.dump(data_config, f, sort_keys=False)

        # app.logger.info(f"data.yaml created at: {data_yaml_path}")
        # with open(data_yaml_path, 'r') as f_read:
        #     app.logger.info(f"data.yaml contents:\n{f_read.read()}")

        # app.logger.info(f"Current CWD: {os.getcwd()}")
        # app.logger.info(f"Absolute path to data.yaml being passed to YOLO: {data_yaml_path}")
        # app.logger.info(f"Absolute path to project output dir for YOLO: {session_folder}")
        original_cwd = os.getcwd()
        os.chdir(session_folder) # Change CWD to where data.yaml and train/val folders are
        app.logger.info(f"Changed CWD for YOLO training to: {os.getcwd()}")


        try:
            model = YOLO('yolov8n-cls.pt')
            results = model.train(
                data='data.yaml',  # YOLO will look for this in the CWD (session_folder)
                epochs=epochs,
                imgsz=640,
                batch=batch_size,
                lr0=learning_rate,
                project='.',        # Outputs will go into CWD/train_run (i.e., session_folder/train_run)
                name='train_run',
                exist_ok=True,
                cache=False
            )                           

            trained_model_path_pt_abs = os.path.join(session_folder, 'train_run', 'weights', 'best.pt')
        finally:
            os.chdir(original_cwd) # Change back CWD
            app.logger.info(f"Restored CWD to: {original_cwd}")
        
        if not os.path.exists(trained_model_path_pt_abs):
            app.logger.error(f"Trained model (best.pt) not found at expected path: {trained_model_path_pt_abs}")
            return jsonify({"error": f"Trained model (best.pt) not found after training. Looked in {trained_model_path_pt_abs}"}), 500

        inference_model = YOLO(trained_model_path_pt_abs)
        inference_model.export(format='tflite')
        exported_tflite_path_abs = trained_model_path_pt_abs.replace('.pt', '.tflite')

        final_model_storage_path = os.path.abspath(MODEL_PATH)
        os.makedirs(os.path.dirname(final_model_storage_path), exist_ok=True)

        if os.path.exists(exported_tflite_path_abs):
            os.rename(exported_tflite_path_abs, final_model_storage_path)
            db_model_path = final_model_storage_path
        else:
            app.logger.error(f"Exported TFLite model not found at: {exported_tflite_path_abs}")
            return jsonify({"error": "Exported TFLite model not found."}), 500

        query = """
            INSERT INTO Training_sessions
            (session_id, class_labels, model_path, created_at)
            VALUES (%s, %s, %s, %s)
        """
        execute_query(query, (session_id, ','.join(class_names), db_model_path, int(datetime.now().timestamp() * 1000)), commit=True)

        final_accuracy = 0.0
        if hasattr(results, 'metrics') and results.metrics:
            final_accuracy = results.metrics.get('metrics/accuracy_top1', 0.0)

        metrics_to_return = {
            "epochs_completed": results.epoch if hasattr(results, 'epoch') else epochs,
            "top1_accuracy": final_accuracy,
        }

        app.logger.info("Training successful.")
        return jsonify({"status": "success", "session_id": session_id, "model_path": db_model_path, "metrics": metrics_to_return}), 200
    except Exception as e:
        app.logger.error(f"Training Error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Download Model ---
@app.route('/download_model', methods=['GET'])
def download_model():
    model_to_download_path = os.path.abspath(MODEL_PATH)
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
    query = "SELECT COUNT(*) as total_modules FROM Module_table"
    result = execute_query(query, fetch_one=True)
    total_modules = result['total_modules'] if result else 0
    return render_template('dashboard.html', total_modules=total_modules)

@app.route('/modules')
def manage_modules():
    query = "SELECT * FROM Module_table"
    modules = execute_query(query)
    return render_template('modules.html', modules=modules)

@app.route('/modules/create', methods=['GET', 'POST'])
def create_module():
    if request.method == 'POST':
        data = request.form
        query = """
            INSERT INTO Module_table
            (id, title, description, created_at, updated_at, link_video)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        args = (
            data.get('id'), data.get('title'), data.get('description'),
            timestamp, timestamp, data.get('link_video')
        )
        execute_query(query, args, commit=True)
        return redirect(url_for('manage_modules'))
    return render_template('create_module.html')

@app.route('/modules/<module_id>/edit', methods=['GET', 'POST'])
def edit_module(module_id):
    if request.method == 'POST':
        data = request.form
        query = """
            UPDATE Module_table
            SET title = %s, description = %s, updated_at = %s, link_video = %s
            WHERE id = %s
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        args = (
            data.get('title'), data.get('description'),
            timestamp, data.get('link_video'), module_id
        )
        execute_query(query, args, commit=True)
        return redirect(url_for('manage_modules'))

    query = "SELECT * FROM Module_table WHERE id = %s"
    module = execute_query(query, (module_id,), fetch_one=True)
    if not module:
        return "Module not found", 404
    return render_template('edit_module.html', module=module)

@app.route('/modules/<module_id>/delete', methods=['POST'])
def delete_module(module_id):
    # Consider deleting associated questions or handling foreign key constraints
    # For example, delete questions first:
    # execute_query("DELETE FROM Question_table WHERE module_id = %s", (module_id,), commit=True)
    query = "DELETE FROM Module_table WHERE id = %s"
    execute_query(query, (module_id,), commit=True)
    return redirect(url_for('manage_modules'))

@app.route('/modules/<module_id>/questions')
def manage_questions(module_id):
    module_query = "SELECT * FROM Module_table WHERE id = %s"
    module = execute_query(module_query, (module_id,), fetch_one=True)
    if not module:
        return "Module not found", 404

    questions_query = "SELECT * FROM Question_table WHERE module_id = %s"
    questions = execute_query(questions_query, (module_id,))

    return render_template('questions.html', module=module, questions=questions)

@app.route('/modules/<module_id>/questions/create', methods=['GET', 'POST'])
def create_question(module_id):
    if request.method == 'POST':
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

    return render_template('create_question.html', module_id=module_id)

@app.route('/questions/<question_id>/edit', methods=['GET', 'POST'])
def edit_question(question_id):
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

    query = "SELECT * FROM Question_table WHERE id = %s"
    question = execute_query(query, (question_id,), fetch_one=True)
    if not question:
        return "Question not found", 404
    return render_template('edit_question.html', question=question)

@app.route('/questions/<question_id>/delete', methods=['POST'])
def delete_question(question_id):
    module_query = "SELECT module_id FROM Question_table WHERE id = %s"
    question = execute_query(module_query, (question_id,), fetch_one=_True)
    if not question:
        # If the question doesn't exist, maybe redirect to a general page
        app.logger.warning(f"Attempted to delete non-existent question or question without module_id: {question_id}")
        return redirect(url_for('manage_modules')) 

    query = "DELETE FROM Question_table WHERE id = %s"
    execute_query(query, (question_id,), commit=True)

    return redirect(url_for('manage_questions', module_id=question['module_id']))

# --- Additional Endpoints ---
@app.route('/upload', methods=['POST'])
def upload_image():
    label = request.form.get('label')
    image = request.files.get('image')

    if not label or not image or not image.filename:
        return jsonify({'error': 'Label and image file are required.'}), 400

    # Secure the label if used as part of a path
    label_folder = os.path.join(UPLOAD_FOLDER, secure_filename(label)) 
    os.makedirs(label_folder, exist_ok=True)

    filename = secure_filename(image.filename)
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
    # Make sure to configure a proper logger for production
    # For development, Flask's default logger with debug=True is often sufficient
    app.run(host='0.0.0.0', port=5000, debug=True)