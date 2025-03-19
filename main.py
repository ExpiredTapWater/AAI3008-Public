import time
import json
import torch
import spacy
import pydub
import random
import whisperx
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from pydub import AudioSegment

# ---------- Fix for RuntimeError: no running event loop ----------
import os
torch.classes.__path__ = []

# ------------------------- WEBPAGE SETUP -------------------------

# Set webpage title (Must be first streamlit code to run!)
st.set_page_config(page_title="AAI3008", initial_sidebar_state="collapsed")

# Change font
with open( "./resources/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
# -------------------------- FOR VAST AI --------------------------
HF_API_KEY = os.getenv("HF_API_KEY") or st.secrets['HF_API_KEY']
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets['GEMINI_API_KEY']

# -------------------------- MODEL SETUP --------------------------



# Model Setup
TRANSCRIPTION_MODEL_NAME = "base"
SPACY_MODEL = "./resources/trained_spacy_model"

# Run Variables
langauge_flag = False

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

# Function to load spacy model with session caching
@st.cache_resource
def load_ner_model():
    
    model = spacy.load(SPACY_MODEL)
    print(f"NER model loaded successfully: {SPACY_MODEL}")

    return model

# Function to load text alignment model
@st.cache_resource
def load_alignment_model():
    model, metadata = whisperx.load_align_model(language_code='en', device=st.session_state.device)
    return model, metadata

# Function to load diarization alignment model
@st.cache_resource
def load_diarization_model():
    model = whisperx.DiarizationPipeline(use_auth_token=HF_API_KEY, device=st.session_state.device)
    return model


# ------------------------ FIRST RUN CODE ------------------------

# Check if this is the first time user loads the webpage.
if "firstrun" not in st.session_state:
    st.session_state.firstrun = True

    # Put any one-time functions here!
    with st.spinner("Please wait..."):
        DEVICE = check_device()
        st.session_state.device = DEVICE
        st.session_state.S2T = load_transcription_model() # Load S2T and store in session
        st.session_state.NER = load_ner_model() # Load NER model and store in session
        st.session_state.ALN, st.session_state.META = load_alignment_model() # Load wav2vec alignment model and store in session
        st.session_state.DIA = load_diarization_model()
        st.session_state.confirm = False # Holds user confirmation state
        st.session_state.language = 'en' # Holds user selected language

        # For chatbot
        st.session_state.history = [] # Holds conversation history
        st.session_state.message = "" # Holds last input text from user
        
        # For downloads
        st.session_state.transcript_ready = False
        st.session_state.masked_transcript_ready = False
        st.session_state.diarization_ready = False

# Fix some nonesence issue
st.session_state.message = ""

# ------------------------ MODEL FUNCTIONS ------------------------

# Original function provided by Dylan
def anonymize_text(text):
    """
    Detects sensitive entities and replaces them with placeholders.
    """
    doc = st.session_state.NER(text)
    new_text = text  # Store modified text
    masked = False # State if a word has been replaced
    
    for ent in doc.ents:
        if ent.label_ in ["NRIC", "PASSPORT_NUM", "EMAIL", "CREDIT_CARD", "BANK_ACCOUNT", "PHONE", "CAR_PLATE"]:  # Add more if needed
            masked = True
            replacement = '[MASK]'  # Can use 'XXXX-XXXX' or something else
            new_text = new_text.replace(ent.text, replacement)

    return new_text, masked

# Modified function for use with muting the audio directly
def find_entities(text):

    doc = st.session_state.NER(text)
    entities = set()  # Use a set to avoid duplicates

    for ent in doc.ents:
        if ent.label_ in ["NRIC", "PASSPORT_NUM", "EMAIL", "CREDIT_CARD", "BANK_ACCOUNT", "PHONE", "CAR_PLATE"]:
            entities.add(ent.text)  # Store sensitive entity
    
    return entities

# Includes fix for double word entities not catching single word phrases
def mute_entities(audio, entities, timestamps):
    mute_regions = []

    # Join words into phrases and check if they match
    for entry in timestamps:
        words = [word_entry["word"] for word_entry in entry["words"]]
        phrase = " ".join(words)  # Create a full phrase

        for entity in entities:
            if entity in phrase:  # Check if any entity is within the phrase
                entity_words = entity.split()
                for i in range(len(words) - len(entity_words) + 1):
                    if words[i:i + len(entity_words)] == entity_words:
                        start_ms = int(entry["words"][i]["start"] * 1000)
                        end_ms = int(entry["words"][i + len(entity_words) - 1]["end"] * 1000)
                        mute_regions.append((start_ms, end_ms))

    # Apply all mute regions at once
    for start_ms, end_ms in sorted(mute_regions, reverse=True):  # Reverse to avoid shifting issues
        silence = AudioSegment.silent(duration=(end_ms - start_ms))
        audio = audio[:start_ms] + silence + audio[end_ms:]

    # Save to a temp file
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_audio.name, format="wav")

    return temp_audio.name  # Return the temp file path

# ---------------------- STREAMLIT FUNCTIONS ----------------------

# Callback function to update state via callback functions (So it freshes on 1st click instead of 2nd)
def confirm_callback():
    st.session_state.confirm = True

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

# -------------------- START OF WEBPAGE CONTENT -------------------

# Introductionary text
st.caption("**AAI3008 Large Language Models Project**")
st.markdown("## :blue[Group 3:] Speech De-identification \n##### ")

# Mutli select options
option_map = {
    1: ":material/mic: Microphone",
    2: ":material/publish: Upload File"}

# Mutli select options
language_map = {
    'auto': ":material/rotate_auto: Auto Detect",
    'en': ":material/language_gb_english: English",
    'zh': ":material/language_chinese_quick: Chinese"}

# Multi selection
selection = st.pills(
    label="**Audio Input Method**",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
    key="input-select",
    default=[1]
)

st.markdown(" ") # Add a small gap

# Prompt user if none is selected
if not selection:
    st.info("Please select an audio input method")

else:

    if selection == 1:

        audio_input = st.audio_input("Use your device's microphone to record audio")
        if audio_input is None:
            st.info("*Record something to continue*")

    elif selection == 2:

        audio_input = st.file_uploader("Upload an existing audio file", type=["wav", "mp3"])
        if audio_input is None:
            st.info("*Upload a file to continue*")
        

    # Preview the audio file
    if audio_input is not None and st.session_state.confirm is not True:
        
        st.markdown(" ") # Add a small gap
        st.audio(audio_input, format="audio/wav", autoplay=False)
        st.markdown(" ") # Add a small gap

        # Multi selection
        language_selection = st.pills(
            label="**Input Language**",
            options=language_map.keys(),
            format_func=lambda option: language_map[option],
            selection_mode="single",
            key="lang-select",
            default=['auto']
        )

        if not language_selection:
            st.info("Please select an input language")

        else:

            st.session_state.language = language_selection

            st.markdown(" ") # Add a small gap
            st.caption("Check audio, then press **Transcribe** to continue")

            # Show confirm button only if not already confirmed
            if not st.session_state.confirm:
                st.button("Transcribe", on_click=confirm_callback)

        # Reset confirmation state if no audio is uploaded
        if audio_input is None:

            # Reset state (Prevents auto confirmation if new input is selected)
            st.session_state.confirm = False
        
# ---------------------- TRANSCRIPTION START  ---------------------

# User has clicked transcribe, continue with further processing
if st.session_state.confirm and audio_input is not None:
    #st.session_state.confirm = False

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_input .getbuffer())
        temp_audio_path = temp_audio.name

    with st.spinner("Transcribing...", show_time=True):

        # Load the temp audio
        audio = whisperx.load_audio(temp_audio_path)

        # Get transcription
        if st.session_state.language != 'auto':
            result = st.session_state.S2T.transcribe(audio, batch_size=1, language=st.session_state.language)
        else:
            result = st.session_state.S2T.transcribe(audio, batch_size=1)
        
        num_speakers = len(result["segments"])
        detected_language = result["language"]

    st.markdown("#### ") # Add gap
    st.caption(f'**Transcribed Text** (Language: {detected_language})')
    for speaker in range(num_speakers):
        st.markdown(f"**Speaker {speaker+1}**")
        st.write_stream(stream_text(result["segments"][speaker]["text"]))

    langauge_flag = True

    # Save as json to pass to chatbot
    with open("transcript.txt", "w") as json_file:
        json.dump(result["segments"], json_file, indent=4)
        st.session_state.transcript_ready = True

    # Checks for incompatible language before continuing to NER
    if result["language"] not in ['en']:
        st.warning("Detected language is not currently supported for NER.")
        langauge_flag = False

# -------------------- ENTITY RECOGNITION START  -------------------

# Run only if language is supported
if langauge_flag:

    mask_flag = False

    alignment_result = whisperx.align(result["segments"], 
                                      st.session_state.ALN, 
                                      st.session_state.META, 
                                      audio, st.session_state.device, return_char_alignments=False)
    
    # Display masked text
    st.markdown(" ") # Add small gap
    st.caption('**Masked Text**')

    # Run NER on the transcribed text
    with st.spinner("Running Audio Modification"):  


        # Load original audio in case there are detected entities
        try: # Very dumb way to ensure audio is wav lol. Later on will be used by mute_entities which guarentees wav output
            temp_file = AudioSegment.from_wav(temp_audio_path)
        except pydub.exceptions.CouldntDecodeError:
            temp_file = AudioSegment.from_mp3(temp_audio_path)

        # Fix the new temp file first in case no matches are found (so entire code chunk below is bypassed)
        new_temp_file = temp_file

        # Iterate over each speaker
        for speaker in range(num_speakers):

            # Get the masked text
            masked_text, masked = anonymize_text(result["segments"][speaker]["text"])

            # If some elements was changed
            if masked:

                # Display masked text, and set flag
                st.markdown(f"**Speaker {speaker+1}**")
                st.write_stream(stream_text(masked_text))
                
                # Get entities list and modify audio
                entities_list = find_entities(result["segments"][speaker]["text"])
                new_temp_file = mute_entities(temp_file, entities_list, alignment_result["segments"])
                temp_file = AudioSegment.from_wav(new_temp_file)

                mask_flag = True

    if mask_flag: # There are words masked, audio modification can be done 

        # Show modified audio
        st.markdown(" ") # Add small gap
        st.caption("**Modified Audio**")
        st.audio(new_temp_file, format="audio/wav")

        # Save as json for user to download later
        with open("masked_transcript.txt", "w") as json_file:
            json.dump(alignment_result["segments"], json_file, indent=4)
            st.session_state.masked_transcript_ready = True
    
    else: # No masking was done, no audio modifcation needs to be done
        st.info("No sensitive entities were detected. Masking and audio modification not needed")

# ------------------- SPEAKER DIARIZATION START  -------------------

if st.session_state.confirm and audio_input is not None and langauge_flag:

    # Reset confirmation status only on last function (which is this one)
    st.session_state.confirm = False

    # Run speaker diarization
    with st.spinner("Running Speaker Diarization"):
        diarize_segments = st.session_state.DIA(audio)
        result = whisperx.assign_word_speakers(diarize_segments, alignment_result)

    # Save as json for user to download later
    with open("diarization.txt", "w") as json_file:
        json.dump(alignment_result["segments"], json_file, indent=4)
        st.session_state.diarization_ready = True

    # Display results
    st.markdown(" ") # Add small gap
    st.caption('**Speaker Diarization**')

    # Get the initial segments from the assigned results
    segments = result.get("segments", [])

    # Define lists for friendly names, icons, and background colors
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
    icons = ["😃", "😎", "🤖", "👻", "🐱", "🐶", "🐵", "🌟", "🔥", "💡"]
    background_colors = [
        "#1F3A93",  # Deep Blue
        "#2C3E50",  # Navy Blue
        "#4D5656",  # Charcoal Gray
        "#7F8C8D",  # Dim Gray
        "#2E4053",  # Dark Slate Blue
        "#212F3C"   # Dark Blue Gray
    ]

    # Get the unique speaker labels
    unique_speakers = sorted({segment.get("speaker", "Unknown Speaker") for segment in segments})

    # Shuffle the names list to assign random names
    random.shuffle(names)

    # Create mappings for speaker -> random friendly name, icon, and background color
    name_map = {speaker: names[i % len(names)] for i, speaker in enumerate(unique_speakers)}
    icon_map = {speaker: icons[i % len(icons)] for i, speaker in enumerate(unique_speakers)}
    bg_color_map = {speaker: background_colors[i % len(background_colors)] for i, speaker in enumerate(unique_speakers)}

    # Display each segment with a colored background using HTML styling
    current_speaker = None
    start_time = None
    end_time = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown Speaker")
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")

        random_name = name_map.get(speaker, speaker)
        icon = icon_map.get(speaker, "🗣️")
        bg_color = bg_color_map.get(speaker, "#1F3A93")

        # If speaker changes, display st.audio for the previous speaker
        if current_speaker and speaker != current_speaker:
            st.audio(temp_audio_path, format="audio/wav", start_time=start_time, end_time=end_time + 1)

        # If new speaker, reset timing
        if speaker != current_speaker:
            start_time = start

        end_time = end  # Update end time

        # Display speaker's text
        st.markdown(
            f"""
            <div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <strong>{icon} {random_name}</strong> ({start:.2f}s - {end:.2f}s)<br>
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

        current_speaker = speaker  # Update current speaker

    # Display the final speaker's st.audio
    if current_speaker:
        st.audio(temp_audio_path, format="audio/wav", start_time=start_time, end_time=end_time + 1)
        

    st.markdown(" ") # Small gap
    st.caption("***End of Speaker Diarization***")

    st.markdown(" ")
    st.page_link("pages/chat.py", label="Go To Chatbot", icon=":material/chat:")
    st.markdown(" ")


# ------------------- DISPLAY DOWNLOAD LINKS  -------------------

if st.session_state.transcript_ready:
    
    with open("transcript.txt", "rb") as file:
        st.download_button(
            label="Download Transcriptions",
            data=file,
            file_name="transcript.txt",
            mime="text/plain",
            icon=":material/download:",
            use_container_width=True

        )

    if st.session_state.masked_transcript_ready:
        with open("masked_transcript.txt", "rb") as file:

            st.download_button(
                label="Download Masked Transcriptions",
                data=file,
                file_name="masked_transcript.txt",
                mime="text/plain",
                icon=":material/download:",
                use_container_width=True
            )

    if st.session_state.diarization_ready:
        with open("diarization.txt", "rb") as file:

            st.download_button(
                label="Download Diarization",
                data=file,
                file_name="diarization.txt",
                mime="text/plain",
                icon=":material/download:",
                use_container_width=True
            )

    

