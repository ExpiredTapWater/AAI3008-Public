import time
import torch
import base64
import streamlit as st
import google.generativeai as genai

# ---------- Fix for RuntimeError: no running event loop ----------
import os
torch.classes.__path__ = []

# ------------------------- WEBPAGE SETUP -------------------------

# Set webpage title (Must be first streamlit code to run!)
st.set_page_config(page_title="Chatbot", initial_sidebar_state="collapsed")
st.markdown('''<style>header {visibility: hidden;}</style>''', unsafe_allow_html=True)

# Change font
with open( "./resources/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# -------------------------- MODEL SETUP --------------------------

# Configure Google AI
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Model config
system_prompt = "You are a helpful chatbot answering questions based on a provided transcript"
model_name = "gemini-2.0-flash"
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 256,
    "response_mime_type": "text/plain",
}

# Load Gemini model
@st.cache_resource
def load_model() -> genai.GenerativeModel:
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_prompt
    )
    return model

# Read and encode the PDF file
def encode(filename):
    with open(filename, "rb") as doc_file:
        return base64.standard_b64encode(doc_file.read()).decode("utf-8")

# Write stream for chatbot
def response_generator(sentence):
    for word in sentence.split():
        yield word + " "
        time.sleep(0.05)

if "history" not in st.session_state:
    st.session_state.history = [] # Holds conversation history
    st.session_state.message = "" # Holds last input text from user

# Load model
model = load_model()

#  ------------------------ Chatbot Start ------------------------

# Introductionary text
st.caption("**AAI3008 Large Language Models Project**")
st.markdown("## :blue[Group 3:] Speech De-identification \n##### ")

st.title("Transcript Chatbot")
st.markdown("Ask the chatbot anything related to audio or transcription of your most recent upload")
st.markdown(" ")

# Read and encode the reference file
try:
    reference = encode("transcript.txt")
    chatbot_ready = True
except FileNotFoundError:
    chatbot_ready = False

# Initalise formatted conversational history
conversation = []

if chatbot_ready:

    # Display chat messages from history on app rerun
    for history in st.session_state.history:
        
        # Get avatar
        if(history["role"] == 'user'):
            avatar = ":material/accessibility_new:"
            role = 'user'
        else:
            avatar = ":material/robot_2:"
            role = 'model'
        
        # Display
        with st.chat_message(history["role"], avatar=avatar):
            st.markdown(history["content"])
            
        # Format history to pass to Chatbot
        conversation.append({"role": role, "parts": history["content"]})

    if st.session_state.message:
        
        # Add new message to dialog
        #st.session_state.dialog += f"User: {st.session_state.message}\n\nAssistant: "
        st.session_state.history.append({"role": "user", "content": st.session_state.message})
        
        # Display user message in chat message container
        with st.chat_message("user", avatar=":material/accessibility_new:"):
            st.markdown(st.session_state.message)
        
        # Display assistant response
        with st.spinner(text="Please Wait"):
            
            # Prompt LLM to get response
                
            # Have to create a new chat each time due to streamlit limitations
            chat = model.start_chat(history=conversation)
            print(f"Full history: {conversation}")
            
            # Get response
            try:
                #response = chat.send_message(st.session_state.message)

                response = chat.send_message([
                            {"mime_type": "text/plain", "data": reference},
                            f"You can reference the provided document if needed to answer this question: '{st.session_state.message}'"])
                
                response.resolve()
                output_text = response.text
                
            except Exception as e:
                print(e)
                output_text = "Sorry, this function has been temporarily disabled due to traffic. Please try again later!"
                
            # Debug
            #print(st.session_state.history)
            
            # Reset message once done
            st.session_state.message = ""
            
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            stream = st.write_stream(response_generator(output_text))

        # Add assistant response to chat history
        st.session_state.history.append({"role": "assistant", "content": output_text})

    st.text_input(label="2", label_visibility="hidden", placeholder="Type Here" ,key="message")

    st.markdown("# ")
    st.divider()
    st.page_link("main.py", label="Return to Homepage", icon=":material/arrow_back:")

else:
    st.info("Please obtain a obtain a transcript from the main page first!")
    st.markdown("# ")
    st.divider()
    st.page_link("main.py", label="Return to Homepage", icon=":material/arrow_back:")