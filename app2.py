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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
import av
import soundfile as sf
import time
import serial
import logging
import platform  # To help list serial ports (optional, for more advanced selection)

logging.basicConfig(level=logging.INFO)

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
.timer {
    font-size: 1.2em;
    font-weight: bold;
    color: #007bff;
    margin-top: 10px;
    text-align: center;
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

def trigger_alert_hardware(port='COM3', baudrate=9600, alert_type='ALERT'):
    try:
        with serial.Serial(port, baudrate, timeout=2) as ser:
            ser.write(f'{alert_type}\n'.encode())
        st.success(f"✅ Hardware alert '{alert_type}' triggered on port {port}.")
    except Exception as e:
        st.warning(f"Could not trigger hardware alert '{alert_type}' on port {port}: {e}")

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
st.markdown("<h1 class='main-header'>🎙️ Live Call Emotion Assistant</h1>", unsafe_allow_html=True)

# --- Hardware Alert Configuration ---
st.sidebar.header("Hardware Alert Configuration")
default_port = 'COM3' if platform.system() == "Windows" else '/dev/ttyACM0' # Common defaults
serial_port = st.sidebar.text_input("Serial Port", default_port)
baud_rate = st.sidebar.number_input("Baud Rate", min_value=9600, value=9600, step=100)

st.sidebar.subheader("Manual Trigger")
if st.sidebar.button("Trigger LED"):
    trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='LED')
if st.sidebar.button("Trigger Buzzer"):
    trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='BUZZER')

# --- Upload Audio Section ---
st.subheader("📁 Upload Audio for Analysis")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    temp_audio_path = Path("temp_uploaded.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(str(temp_audio_path))

    if st.button("Analyze Uploaded Audio"):
        emotion = predict_emotion(str(temp_audio_path))
        if emotion:
            mental_condition, color = map_emotion_to_state(emotion)
            st.markdown(f"<div class='emotion-display' style='background-color: {color};'>{mental_condition.upper()} VOICE</div>", unsafe_allow_html=True)
            if mental_condition == "Abusive":
                play_audio_autoplay("Abusive language detected. Follow protocol.")
                trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='ABUSIVE') # Trigger on abusive
            y, sr = librosa.load(str(temp_audio_path), duration=5)
            time_axis = np.linspace(0, len(y)/sr, num=len(y))
            fig = go.Figure([go.Scatter(x=time_axis, y=y)])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not classify audio.")
        if temp_audio_path.exists():
            temp_audio_path.unlink()

# --- Mic Recording with single toggle and live timer ---
st.markdown("---")
st.subheader("🎤 Record and Analyze Voice")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.recording = False

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.recording:
            self.audio_frames.append(frame)
        return frame

    def start_recording(self):
        self.audio_frames = []
        self.recording = True
        logging.info("Audio recording started.")

    def stop_recording(self):
        self.recording = False
        logging.info("Audio recording stopped.")

    def get_audio_np_array(self):
        if not self.audio_frames:
            logging.warning("No audio frames recorded.")
            return None
        arrays = [frame.to_ndarray(format="s16") for frame in self.audio_frames]
        if arrays:
            combined = np.hstack(arrays).flatten()
            logging.info(f"Combined audio array of shape: {combined.shape}")
            return combined.astype(np.int16)
        else:
            logging.warning("No audio data in frames.")
            return None

audio_processor = AudioProcessor()
recorded_audio_placeholder = st.empty() # Placeholder for audio and duration

webrtc_ctx = webrtc_streamer(
    key="mic_audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=lambda: audio_processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    logging.info("WebRTC context is playing.")
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
        st.session_state.record_start_time = None

    btn_label = "Stop Recording & Analyze" if st.session_state.is_recording else "Start Recording"
    if st.button(btn_label):
        if not st.session_state.is_recording:
            audio_processor.start_recording()
            st.session_state.is_recording = True
            st.session_state.record_start_time = time.time()
            st.success("🔴 Recording started...")
        else:
            audio_processor.stop_recording()
            st.session_state.is_recording = False
            st.success("⏹️ Recording stopped. Analyzing audio...")

            audio_np = audio_processor.get_audio_np_array()

            if audio_np is not None and len(audio_np) > 1000:
                wav_path = "temp_recorded.wav"
                sr = 48000
                sf.write(wav_path, audio_np, sr, subtype='PCM_16')
                duration = librosa.get_duration(path=wav_path)
                duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
                logging.info(f"Saved recorded audio to: {wav_path}, Duration: {duration_str}")

                # Play recorded audio in UI
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
                with recorded_audio_placeholder.container():
                    st.audio(audio_bytes, format="audio/wav")
                    st.markdown(f"<p class='audio-duration'>Duration: {duration_str}</p>", unsafe_allow_html=True)
                logging.info("Displayed recorded audio and duration.")

                emotion = predict_emotion(wav_path)
                if emotion:
                    mental_condition, color = map_emotion_to_state(emotion)
                    st.markdown(f"<div class='emotion-display' style='background-color: {color};'>{mental_condition.upper()} VOICE</div>", unsafe_allow_html=True)
                    if mental_condition == "Abusive":
                        play_audio_autoplay("Abusive language detected. Follow emergency protocol.")
                        trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='ABUSIVE') # Trigger on abusive
                else:
                    st.warning("🤷‍♂️ Unable to detect emotion from recording.")
                    logging.warning("Emotion detection failed for recorded audio.")

                if Path(wav_path).exists():
                    os.remove(wav_path)
                    logging.info(f"Removed temporary file: {wav_path}")
            else:
                st.error("⚠️ No valid audio captured. Please try again.")
                logging.error("No valid audio captured from microphone.")
                recorded_audio_placeholder.empty() # Clear any previous audio

    if st.session_state.is_recording and st.session_state.record_start_time:
        elapsed_time = int(time.time() - st.session_state.record_start_time)
        st.markdown(f"<div class='timer'>⏱️ Recording time: {elapsed_time} second(s)</div>", unsafe_allow_html=True)
else:
    logging.warning("WebRTC context is not playing.")
    st.warning("🔴 Microphone access might not be enabled or the component failed to initialize.")
    recorded_audio_placeholder.empty() # Clear any previous audio