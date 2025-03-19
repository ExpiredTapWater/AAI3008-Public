import os
import time
import queue
import torch
import tempfile
import whisperx
import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pydub import AudioSegment

# ---------- Fix for RuntimeError: no running event loop ----------
import os
torch.classes.__path__ = []

# ------------------------- WEBPAGE SETUP -------------------------

# Set webpage title (Must be first streamlit code to run!)
st.set_page_config(page_title="Live Transcription", initial_sidebar_state="collapsed")
st.markdown('''<style>header {visibility: hidden;}</style>''', unsafe_allow_html=True)

# Change font
with open( "./resources/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# ------------------ REAL TIME THRESHOLD VALUES -------------------
AMPLITUDE_THRESHOLD = 0
SILENT_FRAMES_THRESHOLD = 0

# Check if the page is 'live.py'
if "energy" not in st.query_params:
    st.query_params["energy"] = 5000  # Set default value

if "chunk" not in st.query_params:
    st.query_params["chunk"] = 70  # Set default value

else:
    # Get the query parameters
    query_params = st.query_params

    # Assign 'value' to 100 if the query parameter is missing, otherwise use the provided value
    AMPLITUDE_THRESHOLD = int(query_params.get("energy"))
    SILENT_FRAMES_THRESHOLD = int(query_params.get("chunk"))

# -------------------------- MODEL SETUP --------------------------

# Model Setup
TRANSCRIPTION_MODEL_NAME = "small"

# Function to run on startup
@st.cache_resource
def check_device():

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

# Function to load model with session caching
@st.cache_resource
def load_transcription_model():
    
    COMPUTE = "float16" if st.session_state.device == "cuda" else "float32"

    model = whisperx.load_model(TRANSCRIPTION_MODEL_NAME, 
                                device=st.session_state.device,
                                compute_type=COMPUTE)
    
    print(f"S2T Model loaded successfully: {TRANSCRIPTION_MODEL_NAME}")

    return model

# ------------------------ FIRST RUN CODE ------------------------

# Update type of device
DEVICE = check_device()
st.session_state.device = DEVICE

# Load model
st.session_state.S2T_RT = load_transcription_model()
st.session_state.full_transcript = ""
st.session_state.message = ""

# ---------------------- STREAMLIT FUNCTIONS ----------------------
# Code referenced from: https://github.com/whitphx/streamlit-stt-app

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:

    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")

# Updated code replacing the original deepspeach with WhisperX implementation
def transcribe(audio_segment: AudioSegment, debug: bool = False) -> str:

    if debug:
        save_audio(audio_segment, "debug_audio")

    # Export audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        temp_filename = tmpfile.name
    audio_segment.export(temp_filename, format="wav")
    
    # Load and transcribe the audio file
    audio = whisperx.load_audio(temp_filename)

    # Replaced older API calls with newer transcribe function
    result = st.session_state.S2T_RT.transcribe(audio, batch_size=1, language="en")
    print("Transcription result:", result)
    
    # Extract the transcribed text from segments if available
    if "segments" in result:
        transcript = " ".join([segment["text"] for segment in result["segments"]])
    else:
        transcript = ""
    
    os.remove(temp_filename)
    return transcript

# Calculates the amplitude to determine when to 'chop' the audio
def frame_energy(frame):

    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16).astype(np.int32)
    return np.sqrt(np.mean(samples**2))

# As this works in chunks to process, this function appends each frame (because of how webrtc works) to chunk
def add_frame_to_chunk(audio_frame, sound_chunk: AudioSegment) -> AudioSegment:

    sound = AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    return sound_chunk + sound
# Process a list of audio frames, update the sound chunk, and count consecutive low-energy frames.
def process_audio_frames(audio_frames, sound_chunk: AudioSegment, silence_frames, energy_threshold):

    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)
        energy = frame_energy(audio_frame)
        if energy < energy_threshold:
            silence_frames += 1
        else:
            silence_frames = 0
    return sound_chunk, silence_frames

def handle_silence(sound_chunk: AudioSegment, silence_frames, silence_frames_threshold):

    """
    If enough silence is detected, transcribe the current audio chunk,
    append the text to the transcript stored in session state,
    and reset the audio chunk.
    """

    if silence_frames >= silence_frames_threshold:
        if len(sound_chunk) > 0:
            text = transcribe(sound_chunk)
            st.session_state["transcript"] += text + " "
            sound_chunk = AudioSegment.silent(duration=0)
            silence_frames = 0
    return sound_chunk, silence_frames
# When no audio frames are received within the timeout, process any pending audio.
def handle_queue_empty(sound_chunk: AudioSegment):

    if len(sound_chunk) > 0:
        text = transcribe(sound_chunk)
        st.session_state["transcript"] += text + " "
        sound_chunk = AudioSegment.silent(duration=0)
    return sound_chunk


def app_sst(status_indicator, text_output, timeout=3, energy_threshold=AMPLITUDE_THRESHOLD, silence_frames_threshold=SILENT_FRAMES_THRESHOLD):

    """
    The main function for real-time speech-to-text.
    While the integrated webrtc button is active (audio_receiver is True), the app
    processes audio frames and updates the transcript stored in session state.
    When the button is toggled off (audio_receiver becomes False), the final transcript is displayed.
    """

    with st.spinner("Initialising WebRTC..."):
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            #rtc_configuration={"iceServers": []},  # Uncomment to speed up for LOCAL use only!!
        )

    sound_chunk = AudioSegment.silent(duration=0)
    silence_frames = 0

    while True:
        if webrtc_ctx.audio_receiver:
            status_indicator.write("**Status:** Listening")
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                status_indicator.write("**Status:** No frame arrived.")
                sound_chunk = handle_queue_empty(sound_chunk)
                continue

            sound_chunk, silence_frames = process_audio_frames(
                audio_frames, sound_chunk, silence_frames, energy_threshold
            )
            sound_chunk, silence_frames = handle_silence(
                sound_chunk, silence_frames, silence_frames_threshold
            )
            text_output.write(st.session_state["transcript"])
        else:
            status_indicator.write("**Status:** Not Ready")
            if len(sound_chunk) > 0:
                text = transcribe(sound_chunk)
                st.session_state["transcript"] += text + " "
            text_output.write(st.session_state["transcript"])
            break

def main():
    st.caption("**AAI3008 Large Language Models Project**")
    st.markdown("## :blue[Group 3:] Speech De-identification \n##### ")

    st.title("Real Time Transcription")
    st.markdown(" ")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="**Model**", value=TRANSCRIPTION_MODEL_NAME,help="size of Whisper model used", label_visibility="visible")
    col2.metric(label="**Energy Threshold**", value=AMPLITUDE_THRESHOLD,
                help="Determines if a frame is considered speech or silence", label_visibility="visible")
    col3.metric(label="**Chunk Threshold**", value=SILENT_FRAMES_THRESHOLD
                ,help="Number of silent frames needed before a chunk is cut and sent to be processed", label_visibility="visible")

    st.markdown(" ")
    # Initialize transcript in session state if not already set.
    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ""

    # Need to reserve a slot to update status, if not there will be multiple overlapping copies
    status_indicator = st.empty()

    st.markdown(" ")
    st.caption("**Transcription**")
    text_output = st.empty()
    app_sst(status_indicator, text_output)

main()

st.markdown("# ")
st.divider()
st.page_link("main.py", label="Return to Homepage", icon=":material/arrow_back:")