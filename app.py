import streamlit as st
import numpy as np
import plotly.graph_objs as go
import librosa
from predict_emotion import predict_emotion
from gtts import gTTS
import tempfile
import base64
import os
from pathlib import Path

# --- Custom CSS ---
custom_css = """
<style>
body {
    background-color: #f8f9fa;
    color: #333;
    font-family: 'Inter', sans-serif;
}
.main-header {
    font-size: 3em;
    color: #007bff;
    text-align: center;
    margin-bottom: 25px;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.emotion-display {
    color: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px;
    font-size: 1.5em;
    text-align: center;
    font-weight: bold;
}
.audio-duration {
    font-size: 1em;
    color: #6c757d;
    margin-top: 5px;
    text-align: center;
}
</style>
"""

def play_audio_autoplay(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(text=text, lang='en')
        tts.save(tmpfile.name)
        audio_path = tmpfile.name

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    os.remove(audio_path)

    audio_html = f"""
    <audio id="tts_audio" autoplay>
      <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3" />
    </audio>
    <script>
      var audio = document.getElementById('tts_audio');
      audio.play().catch(function(e) {{
        console.log('Autoplay prevented:', e);
      }});
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def map_emotion_to_state(emotion):
    colors = {
        "Abusive": "#dc3545",
        "Stressful": "#ffc107",
        "Prank": "#28a745",
        "Painful": "#007bff",
        "Normal": "#6c757d",
        "Drunk": "#6f42c1"
    }
    emotion_to_condition_map = {
        "angry": "Abusive",
        "fearful": "Stressful",
        "happy": "Prank",
        "sad": "Painful",
        "calm": "Normal",
        "neutral": "Normal",
        "disgust": "Abusive"
    }
    condition = emotion_to_condition_map.get(emotion, "Unknown")
    color = colors.get(condition, "#343a40")
    return condition, color

st.set_page_config(page_title="SER Assistant", page_icon="🎙️", layout="centered")
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>🎙️ Uploaded Voice Analysis</h1>", unsafe_allow_html=True)

# --- Upload Audio Section ---
st.subheader("📁 Upload Audio for Analysis")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    temp_audio_path = Path("temp_uploaded_audio.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio(str(temp_audio_path), format="audio/wav")

    if st.button("Analyze Uploaded Audio"):
        with st.spinner("Analyzing uploaded audio..."):
            emotion = predict_emotion(str(temp_audio_path))
            
            if emotion:
                mental_condition, color = map_emotion_to_state(emotion)
                st.markdown(
                    f"<div class='emotion-display' style='background-color: {color};'>{mental_condition.upper()} VOICE DETECTED</div>", 
                    unsafe_allow_html=True
                )
                play_audio_autoplay(f"Detected emotion: {emotion}. You sound {mental_condition.lower()}.")
                
                try:
                    y, sr = librosa.load(str(temp_audio_path), sr=None, duration=5)
                    time_axis = np.linspace(0, len(y)/sr, num=len(y))
                    fig = go.Figure([go.Scatter(x=time_axis, y=y, line=dict(color=color))])
                    fig.update_layout(
                        title='Audio Waveform',
                        xaxis_title='Time (s)',
                        yaxis_title='Amplitude',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#333',
                        height=250,
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not visualize waveform: {e}")
            else:
                st.warning("🤷‍♂️ Unable to detect emotion from the uploaded audio. Please try another file.")
        
    if temp_audio_path.exists():
        temp_audio_path.unlink()
