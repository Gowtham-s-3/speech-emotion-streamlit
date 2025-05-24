import joblib
import numpy as np
from utils.features import extract_features

def load_model_and_encoder():
    model = joblib.load("model/svm_emotion_model.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    return model, encoder

def predict_emotion(audio_path):
    model, encoder = load_model_and_encoder()
    features = extract_features(audio_path)
    if features is not None:
        features = np.array(features).reshape(1, -1)
        pred_encoded = model.predict(features)[0]
        emotion = encoder.inverse_transform([pred_encoded])[0]
        return emotion
    return None
