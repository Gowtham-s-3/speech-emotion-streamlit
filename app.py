import streamlit as st
import numpy as np
import plotly.graph_objs as go
import librosa
from predict_emotion import predict_emotion  # Assuming this file exists
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
import platform
import serial.tools.list_ports

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

def trigger_alert_hardware(port, baudrate, alert_type='ALERT'):
    ser = None
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        ser.write(f'{alert_type}\n'.encode())
        st.success(f"✅ Hardware alert '{alert_type}' sent to {port}.")
        logging.info(f"Sent serial command: {alert_type} to {port}")
    except serial.SerialException as e:
        error_message = f"❌ Serial port error on {port}: {e}. "
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            error_message += "The port might be in use or permissions are denied."
        elif "FileNotFoundError" in str(e) or "could not open port" in str(e):
            error_message += "The specified port was not found."
        st.error(error_message)
        logging.error(f"Serial port error on {port}: {e}")
    except Exception as e:
        st.warning(f"⚠️ Could not trigger alert '{alert_type}' on {port}: {e}")
        logging.error(f"Error triggering alert on {port}: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()

def check_all_com_ports():
    """Lists all available COM ports and their descriptions."""
    ports = serial.tools.list_ports.comports()
    if ports:
        st.subheader("Available Serial Ports:")
        for port, desc, hwid in sorted(ports):
            st.write(f"- **{port}**: {desc} ({hwid})")
    else:
        st.info("No serial ports found on this system.")

def map_emotion_to_state(emotion):
    colors = {
        "Abusive": "#dc3545", "Stressful": "#ffc107", "Prank": "#28a745",
        "Painful": "#007bff", "Normal": "#6c757d", "Drunk": "#6f42c1"
    }
    emotion_to_condition_map = {
        "angry": "Abusive", "fearful": "Stressful", "happy": "Prank",
        "sad": "Painful", "calm": "Normal", "neutral": "Normal",
        "disgust": "Abusive"
    }
    condition = emotion_to_condition_map.get(emotion, "Unknown")
    color = colors.get(condition, "#343a40")
    return condition, color

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="SER Assistant", page_icon="🎙️", layout="centered")
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>🎙️ Live Call Emotion Assistant</h1>", unsafe_allow_html=True)

# --- Hardware Alert Configuration in Sidebar ---
st.sidebar.header("Hardware Alert Configuration")

ports = serial.tools.list_ports.comports()
available_ports = [port.device for port in ports]
if not available_ports:
    available_ports = ["No COM ports found"]

selected_port = st.sidebar.selectbox("Select Serial Port", available_ports)
serial_port = selected_port if selected_port != "No COM ports found" else None
baud_rate = st.sidebar.number_input("Baud Rate", min_value=9600, value=9600, step=100)

st.sidebar.subheader("Manual Trigger")
if st.sidebar.button("Trigger LED (D6)", key="led_button") and serial_port:
    trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='LED')
elif st.sidebar.button("Trigger LED (D6)", key="led_warning") and not serial_port:
    st.sidebar.warning("Please select a serial port.")

if st.sidebar.button("Trigger Buzzer (D7)", key="buzzer_button") and serial_port:
    trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='BUZZER')
elif st.sidebar.button("Trigger Buzzer (D7)", key="buzzer_warning") and not serial_port:
    st.sidebar.warning("Please select a serial port.")

st.sidebar.subheader("Debug")
if st.sidebar.button("Check Available COM Ports", key="check_ports_button"):
    check_all_com_ports()

# --- Upload Audio Section ---
st.subheader("📁 Upload Audio for Analysis")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    temp_audio_path = Path("temp_uploaded.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(str(temp_audio_path))

    if st.button("Analyze Uploaded Audio"):
        with st.spinner("Analyzing uploaded audio..."):
            emotion = predict_emotion(str(temp_audio_path))
            if emotion:
                mental_condition, color = map_emotion_to_state(emotion)
                st.markdown(f"<div class='emotion-display' style='background-color: {color};'>{mental_condition.upper()} VOICE</div>", unsafe_allow_html=True)
                if mental_condition == "Abusive" and serial_port:
                    play_audio_autoplay("Abusive language detected. Alerting hardware.")
                    trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='ABUSIVE')
                elif mental_condition == "Abusive" and not serial_port:
                    play_audio_autoplay("Abusive language detected.")
                    st.warning("Serial port not selected for hardware alert.")

                try:
                    y, sr = librosa.load(str(temp_audio_path), duration=5)
                    time_axis = np.linspace(0, len(y)/sr, num=len(y))
                    fig = go.Figure([go.Scatter(x=time_axis, y=y, line=dict(color=color))])
                    fig.update_layout(
                        title='Audio Waveform',
                        xaxis_title='Time (s)',
                        yaxis_title='Amplitude',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#333'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not visualize waveform: {e}")
            else:
                st.warning("Could not classify audio.")
        if temp_audio_path.exists():
            temp_audio_path.unlink()

# --- Live Mic Recording and Analysis Section ---
st.markdown("---")
st.subheader("🎤 Live Record and Analyze Voice")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.recording = False

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.recording:
            self.audio_frames.append(frame)
        return frame

    async def recv_queued(self, frames):
        for frame in frames:
            if self.recording:
                self.audio_frames.append(frame)
        return frames

    def start_recording(self):
        self.audio_frames = []
        self.recording = True
        logging.info("Live audio recording started.")

    def stop_recording(self):
        self.recording = False
        logging.info("Live audio recording stopped.")
        return self.get_audio_np_array()

    def get_audio_np_array(self):
        if not self.audio_frames:
            logging.warning("No live audio frames recorded.")
            return None
        arrays = [frame.to_ndarray(format="s16") for frame in self.audio_frames]
        if arrays:
            combined = np.hstack(arrays).flatten()
            logging.info(f"Combined live audio array shape: {combined.shape}")
            return combined.astype(np.int16)
        else:
            logging.warning("No live audio data in frames after conversion.")
            return None

audio_processor = AudioProcessor()
live_recorded_audio_placeholder = st.empty()
live_analysis_placeholder = st.empty()
live_timer_placeholder = st.empty()

webrtc_ctx = webrtc_streamer(
    key="live_mic_audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=lambda: audio_processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    logging.info("Live WebRTC context is playing.")
    if "live_is_recording" not in st.session_state:
        st.session_state.live_is_recording = False
        st.session_state.live_record_start_time = None

    live_btn_label = "Stop Live Recording & Analyze" if st.session_state.live_is_recording else "Start Live Recording"
    if st.button(live_btn_label):
        if not st.session_state.live_is_recording:
            audio_processor.start_recording()
            st.session_state.live_is_recording = True
            st.session_state.live_record_start_time = time.time()
            st.success("🔴 Live recording started...")
            live_recorded_audio_placeholder.empty()
            live_analysis_placeholder.empty()
        else:
            audio_np = audio_processor.stop_recording()
            st.session_state.live_is_recording = False
            st.success("⏹️ Live recording stopped. Analyzing audio...")
            live_timer_placeholder.empty()

            if audio_np is not None and len(audio_np) > 1000:
                wav_path = "temp_live_recorded.wav"
                sr = 48000
                sf.write(wav_path, audio_np, sr, subtype='PCM_16')
                duration = librosa.get_duration(path=wav_path)
                duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
                logging.info(f"Saved live recorded audio to: {wav_path}, Duration: {duration_str}")

                with live_recorded_audio_placeholder.container():
                    st.audio(wav_path, format="audio/wav")
                    st.markdown(f"<p class='audio-duration'>Duration: {duration_str}</p>", unsafe_allow_html=True)

                with st.spinner("Predicting emotion..."):
                    emotion = predict_emotion(wav_path)

                with live_analysis_placeholder.container():
                    if emotion:
                        mental_condition, color = map_emotion_to_state(emotion)
                        st.markdown(f"<div class='emotion-display' style='background-color: {color};'>{mental_condition.upper()} VOICE</div>", unsafe_allow_html=True)
                        if mental_condition == "Abusive" and serial_port:
                            play_audio_autoplay("Abusive language detected from live recording. Alerting hardware.")
                            trigger_alert_hardware(port=serial_port, baudrate=baud_rate, alert_type='ABUSIVE')
                        elif mental_condition == "Abusive" and not serial_port:
                            play_audio_autoplay("Abusive language detected from live recording.")
                            st.warning("Serial port not selected for hardware alert.")
                    else:
                        st.warning("🤷‍♂️ Unable to detect emotion from live recording.")
                        logging.warning("Emotion detection failed for live recorded audio.")

                if Path(wav_path).exists():
                    os.remove(wav_path)
                    logging.info(f"Removed temporary file: {wav_path}")
            else:
                st.error("⚠️ No valid live audio captured. Please try again.")
                logging.error("No valid live audio captured from microphone.")
                live_recorded_audio_placeholder.empty()
                live_analysis_placeholder.empty()

    if st.session_state.live_is_recording and st.session_state.live_record_start_time:
        elapsed_time = int(time.time() - st.session_state.live_record_start_time)
        live_timer_placeholder.markdown(f"<div class='timer'>⏱️ Live recording time: {elapsed_time} second(s)</div>", unsafe_allow_html=True)
    else:
        live_timer_placeholder.empty()

else:
    logging.warning("Live WebRTC context is not playing.")
    st.warning("🔴 Microphone access might not be enabled. Please allow in your browser.")
    live_recorded_audio_placeholder.empty()
    live_analysis_placeholder.empty()
    live_timer_placeholder.empty()