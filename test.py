import librosa
import numpy as np
import joblib

def extract_features(audio_path):
    """Extracts MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, duration=3, offset=0.5) # Adjust duration/offset if needed
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None
    return mfccs_scaled

if __name__ == "__main__":
    # Load the trained model and label encoder
    loaded_model = joblib.load('svm_emotion_model.pkl')
    loaded_le = joblib.load('label_encoder.pkl')

    # Path to the audio file you want to test
    test_audio_path = 'C:/Users/Admin/Downloads/emotion_project/sample-wav-files-sample3.wav'  # <--- REPLACE WITH YOUR AUDIO FILE PATH

    # Extract features from the test audio
    test_features = extract_features(test_audio_path)

    if test_features is not None:
        # Reshape features to match the model's input shape
        test_features = test_features.reshape(1, -1)

        # Predict the emotion
        predicted_label_encoded = loaded_model.predict(test_features)

        # Decode the predicted label
        predicted_emotion = loaded_le.inverse_transform(predicted_label_encoded)[0]

        print(f"Predicted emotion for '{test_audio_path}': {predicted_emotion}")
    else:
        print("Could not process the test audio file.")