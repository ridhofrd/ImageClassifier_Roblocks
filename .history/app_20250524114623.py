from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_mysqldb import MySQL
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from utils.trainer import train_and_export_model

app = Flask(__name__)
CORS(app)

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'roblock_module'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi helper untuk query database
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

#----------------API Image Classifier-------------------#
@app.route('/train', methods=['POST'])
def train():
    try:
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)

        class_labels = request.form.getlist('class_label')
        epochs = int(request.args.get('epochs', 10))
        batch_size = int(request.args.get('batch_size', 32))
        learning_rate = float(request.args.get('learning_rate', 0.001))
        if not class_labels:
            return {"error": "No class labels provided"}, 400

        class_names = set(class_labels)
        image_paths = []
        labels = []

        files = request.files.getlist('images')
        if len(files) == 0:
            return {"error": "No images provided"}, 400

        image_index = 0
        for label in class_labels:
            label_folder = os.path.join(session_folder, label)
            os.makedirs(label_folder, exist_ok=True)
            for _ in range(len(files) // len(class_labels)):
                if image_index >= len(files):
                    break
                file = files[image_index]
                file_path = os.path.join(label_folder, file.filename)
                file.save(file_path)
                image_paths.append(file_path)
                labels.append(label)
                image_index += 1

        if len(image_paths) < len(class_names) * 2:
            return {"error": "Insufficient images for training"}, 400

        generate_annotations(image_paths, labels)

        train_images, val_images, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        train_folder = os.path.join(session_folder, 'train')
        val_folder = os.path.join(session_folder, 'val')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        for img_path, label in zip(train_images, train_labels):
            dest_folder = os.path.join(train_folder, label)
            os.makedirs(dest_folder, exist_ok=True)
            os.rename(img_path, os.path.join(dest_folder, os.path.basename(img_path)))
            txt_path = img_path.replace('.jpg', '.txt')
            if os.path.exists(txt_path):
                os.rename(txt_path, os.path.join(dest_folder, os.path.basename(txt_path)))
        for img_path, label in zip(val_images, val_labels):
            dest_folder = os.path.join(val_folder, label)
            os.makedirs(dest_folder, exist_ok=True)
            os.rename(img_path, os.path.join(dest_folder, os.path.basename(img_path)))
            txt_path = img_path.replace('.jpg', '.txt')
            if os.path.exists(txt_path):
                os.rename(txt_path, os.path.join(dest_folder, os.path.basename(txt_path)))

        data_config = {
            'train': train_folder,
            'val': val_folder,
            'nc': len(class_names),
            'names': list(class_names)
        }
        with open('data.yaml', 'w') as f:
            yaml.dump(data_config, f)

        model = YOLO('yolov8n.pt')
        model.train(
            data='data.yaml',
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            lr0=learning_rate
        )
        model.export(format='tflite')

        return {"status": "success"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

# ---------------- API MODULE DAN QUIZ UNTUK APLIKASI ANDROID ----------------
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

# ---------------- WEB INTERFACE MODULE DAN QUIZ ----------------
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
        
        module_query = "SELECT module_id FROM Question_table WHERE id = %s"
        question = execute_query(module_query, (question_id,), fetch_one=True)
        return redirect(url_for('manage_questions', module_id=question['module_id']))
    
    query = "SELECT * FROM Question_table WHERE id = %s"
    question = execute_query(query, (question_id,), fetch_one=True)
    if not question:
        return "Question not found", 404
    return render_template('edit_question.html', question=question)

@app.route('/questions/<question_id>/delete', methods=['POST'])
def delete_question(question_id):
    module_query = "SELECT module_id FROM Question_table WHERE id = %s"
    question = execute_query(module_query, (question_id,), fetch_one=True)
    
    query = "DELETE FROM Question_table WHERE id = %s"
    execute_query(query, (question_id,), commit=True)
    
    return redirect(url_for('manage_questions', module_id=question['module_id']))

# ---------------- FITUR UPLOAD, TRAIN, DAN UNDUH MODEL ----------------
@app.route('/upload', methods=['POST'])
def upload_image():
    label = request.form.get('label')
    image = request.files.get('image')

    if not label or not image:
        return jsonify({'error': 'Label and image are required.'}), 400

    label_folder = os.path.join(UPLOAD_FOLDER, label)
    os.makedirs(label_folder, exist_ok=True)

    filename = secure_filename(image.filename)
    image_path = os.path.join(label_folder, filename)
    image.save(image_path)

    return jsonify({'message': f'Image saved to {image_path}'}), 200

@app.route('/train_export_model', methods=['POST'])
def train_export_model():
    print("🔥 Train endpoint hit!")
    try:
        train_and_export_model()
        return jsonify({
            "message": "Model trained successfully.",
            "model_path": "model/color_classifier.tflite"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/download-model', methods=['GET'])
def download_model():
    model_path = os.path.join('model', 'color_classifier.tflite')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route("/ping", methods=["GET"])
def ping():
    print("📡 Received /ping")
    return "pong!"

@app.route('/test', methods=['GET'])
def test():
    return "Server is running!"

if __name__ == '__main__':
    app.run(debug=True)
