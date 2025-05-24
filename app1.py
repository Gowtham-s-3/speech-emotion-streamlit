from flask import Flask, request, jsonify
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

# Load your trained model
model = load_model('emotion_model.h5')

# Define emotion labels (adjust as per your model)
emotion_labels = ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad']

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio_file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        features = extract_features(file_path)

        if features is None:
            return jsonify({'error': 'Could not process audio file'}), 500

        features = np.expand_dims(features, axis=0)  # Model expects batch
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]

        # Optionally delete the file after processing
        os.remove(file_path)

        return jsonify({'emotion': predicted_emotion})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
