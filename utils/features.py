# utils/features.py

import librosa
import numpy as np

def extract_features(audio_file, sr=None, offset=0.5, duration=3):
    """
    Extracts MFCC features from an audio file.

    Args:
        audio_file (str): Path to the audio file.
        sr (int, optional): Sampling rate to load audio. If None, librosa's default is used.
        offset (float): Start time offset in seconds.
        duration (float): Duration to load in seconds.

    Returns:
        numpy.ndarray: Mean of the MFCC features across time, or None if an error occurs.
    """
    try:
        y, sr = librosa.load(audio_file, sr=sr, offset=offset, duration=duration)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

if __name__ == '__main__':
    # Example usage (you'd need a test audio file)
    test_audio = '../assets/audio/sample1.wav'  # Adjust path as needed
    features = extract_features(test_audio)
    if features is not None:
        print("Extracted features shape:", features.shape)
        print("Sample features:", features[:5])
    else:
        print(f"Could not extract features from {test_audio}")