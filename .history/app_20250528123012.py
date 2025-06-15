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

app = Flask(__name__)
CORS(app)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'roblock_module'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Upload and Model Folders
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/model.tflite' # Path for the final trained model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper function for database queries
def execute_query(query, args=None, fetch_one=False, commit=False):
    cur = mysql.connection.cursor()
    cur.execute(query, args)
    if commit:
        mysql.connection.commit()
    if fetch_one:
        result = cur.fetchone()
    else:
        result = cur.fetchall()
    cur.close()
    return result

# ---------------- API Image Classifier ----------------
@app.route('/train', methods=['POST'])
def train():
    # original_cwd = os.getcwd() # No longer needed if not changing CWD
    session_id = str(uuid.uuid4())
    # session_folder will now be D:\Lab-Roblocks\Backend_Roblocks\uploads\<session_id>
    session_folder = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    os.makedirs(session_folder, exist_ok=True)
    app.logger.info(f"Session folder created: {session_folder}") # Will show lowercase 'uploads'

    try:
        # ... (rest of your initial setup, getting class_labels, epochs, etc.) ...

        # train_base_dir will be D:\Lab-Roblocks\Backend_Roblocks\uploads\<session_id>\train
        train_base_dir = os.path.join(session_folder, 'train')
        # val_base_dir will be D:\Lab-Roblocks\Backend_Roblocks\uploads\<session_id>\val
        val_base_dir = os.path.join(session_folder, 'val')
        # ... (create these directories and class subdirectories, move images) ...
        app.logger.info(f"Train base directory: {train_base_dir}")
        app.logger.info(f"Validation base directory: {val_base_dir}")


        # data.yaml uses ABSOLUTE paths for train and val
        data_config = {
            'train': os.path.abspath(train_base_dir),
            'val': os.path.abspath(val_base_dir),
            'nc': len(class_names),
            'names': class_names
        }
        # data_yaml_path will be D:\Lab-Roblocks\Backend_Roblocks\uploads\<session_id>\data.yaml
        data_yaml_path = os.path.join(session_folder, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)

        app.logger.info(f"data.yaml created at: {data_yaml_path}")
        with open(data_yaml_path, 'r') as f_read:
            app.logger.info(f"data.yaml contents:\n{f_read.read()}")

        app.logger.info(f"Current CWD (should be your project root): {os.getcwd()}")
        app.logger.info(f"Absolute path to data.yaml being passed to YOLO: {data_yaml_path}")
        app.logger.info(f"Absolute path to project output dir for YOLO: {session_folder}")

        model = YOLO('yolov8n-cls.pt')

        results = model.train(
            data=data_yaml_path,      # Absolute path to data.yaml (now using lowercase 'uploads')
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            lr0=learning_rate,
            project=session_folder,   # Absolute path for project (now using lowercase 'uploads')
            name='train_run',
            exist_ok=True,
            cache=False               # Explicitly disable caching to be sure
        )

        # trained_model_path_pt_abs will use lowercase 'uploads'
        trained_model_path_pt_abs = os.path.join(session_folder, 'train_run', 'weights', 'best.pt')
        

        if not os.path.exists(trained_model_path_pt_abs): # Check absolute path
            app.logger.error(f"Trained model (best.pt) not found at expected path: {trained_model_path_pt_abs}")
            # Check if the path exists without abspath, just in case CWD assumption is tricky for this check
            if os.path.exists(trained_model_path_pt):
                 app.logger.info(f"However, it was found at relative path: {trained_model_path_pt} from CWD {os.getcwd()}")
                 trained_model_path_pt_abs = os.path.join(os.getcwd(), trained_model_path_pt) # Reconstruct absolute
            else:
                return jsonify({"error": f"Trained model (best.pt) not found after training. Looked in {trained_model_path_pt_abs}"}), 500

        inference_model = YOLO(trained_model_path_pt_abs)
        inference_model.export(format='tflite')
        # Exported TFLite model will be in the same directory as best.pt
        exported_tflite_path_abs = trained_model_path_pt_abs.replace('.pt', '.tflite')

        final_model_storage_path = os.path.abspath(MODEL_PATH) # Global MODEL_PATH
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



@app.route('/download_model', methods=['GET'])
def download_model():
    model_to_download_path = os.path.abspath(MODEL_PATH)
    if os.path.exists(model_to_download_path):
        return send_file(model_to_download_path, as_attachment=True)
    return jsonify({"error": "Model not found"}), 404

# ---------------- API Module and Quiz for Android App ----------------
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

# ---------------- Web Interface Module and Quiz ----------------
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
            data['id'], data['title'], data['description'],
            timestamp, timestamp, data['link_video']
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
            data['title'], data['description'],
            timestamp, data['link_video'], module_id
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
            data['id'], module_id, data['question_text'],
            data['option_a'], data['option_b'], data['option_c'], data['option_d'],
            data['correct_answer'], timestamp, timestamp
        )
        execute_query(query, args, commit=True)
        return redirect(url_for('manage_questions', module_id=module_id))

    return render_template('create_question.html', module_id=module_id)

@app.route('/questions/<question_id>/edit', methods=['GET', 'POST'])
def edit_question(question_id):
    # Fetch module_id first to redirect correctly after edit
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
            data['question_text'], data['option_a'], data['option_b'],
            data['option_c'], data['option_d'], data['correct_answer'],
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
    question = execute_query(module_query, (question_id,), fetch_one=True)
    if not question: # Should not happen if question_id is valid
        return redirect(url_for('manage_modules')) # Fallback redirect

    query = "DELETE FROM Question_table WHERE id = %s"
    execute_query(query, (question_id,), commit=True)

    return redirect(url_for('manage_questions', module_id=question['module_id']))

# ---------------- Additional Endpoints ----------------
@app.route('/upload', methods=['POST'])
def upload_image():
    label = request.form.get('label')
    image = request.files.get('image')

    if not label or not image or not image.filename:
        return jsonify({'error': 'Label and image file are required.'}), 400

    label_folder = os.path.join(UPLOAD_FOLDER, secure_filename(label)) # Secure the label if used as folder name
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
    app.run(host='0.0.0.0', port=5000, debug=True)