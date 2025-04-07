# api/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Untuk CORS
import os
from werkzeug.utils import secure_filename
from utils.trainer import train_and_export_model

app = Flask(__name__)

# Menambahkan CORS
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.route('/train', methods=['POST'])
def train():
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

