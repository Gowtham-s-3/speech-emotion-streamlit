import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import librosa

# === MODEL PATH CONFIGURATION (Top of file) ===
MODEL_PATH = 'model/svm_emotion_model.pkl'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'

# === Flask Setup ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# === Create upload folder if it doesn't exist ===
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load the model and label encoder ===
clf_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# === Allowed file type check ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Feature Extraction Function ===
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        features = np.hstack((mfccs_mean, mfccs_std)).reshape(1, -1)
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# === Routes ===
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        features = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        try:
            prediction = clf_model.predict(features)
            emotion = label_encoder.inverse_transform(prediction)[0]
            os.remove(filepath)
            return jsonify({'emotion': emotion})
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only .wav files are allowed'}), 400

# === Run Server ===
if __name__ == '__main__':
    print(f"Flask running at http://127.0.0.1:5000")
    app.run(debug=True)
