# AAI3008
Repository for AAI3008 Large Language Models - Group Project

## Hosting on Vast AI:

### 1. Visit [Vast.Ai](https://vast.ai/) and login
### 2. Find our custom template [here](https://cloud.vast.ai?ref_id=186058&template_id=d5b3ed5a532d9765590e9fc7808f601a)
![image 1](https://i.postimg.cc/jq7Jw5CD/1.jpg)
- The template is setup to do the following, and can be replicated by:
    1. Add default streamlit port (8501) as internal port to be mapped by Vast.AI later
    2. Add required API keys as environment variables
    3. Sets minumum disk space to 64GB
    4. Add startup script:
        - Clones the public repo `git clone https://github.com/ExpiredTapWater/AAI3008-Public` 
        - Change directory, activate venv, install packages, and runs streamlit in the background
        `cd /workspace/AAI3008-Public && source /venv/main/bin/activate && pip install -r requirements.txt && streamlit run main.py &`
        - Run default entrypoint file `entrypoint.sh`
### 3. Wait around 5 mins
- You can check instance log to know when streamlit is up.
    ![image 2](https://i.postimg.cc/BnM89P7V/2.jpg)

### 4. Check the assigned external IP and port number
![image 3](https://i.postimg.cc/wBM16xST/3.jpg)

### 5. (Optional) Enable HTTPS tunnel
- Some features such as WebRTC for real time transcriptions requires a https connection
- Press "OPEN"
- Click "Tunnels" on the left side
- Enter `http://localhost:8501` and click "Create New Tunnel"
![image 4](https://i.postimg.cc/rw8dPNj4/4.jpg)
- Acess streamlit through the generated URL
![image 5](https://i.postimg.cc/YqNGRs6k/5.jpg)

**If you are unable to access the link, simply swap your DNS in Windows to anybody else eg. Google (8.8.8.8)**

## Program Screenshots (For future reference)
https://reaching-minus-dos-inches.trycloudflare.com

### Main Page
![1](https://i.postimg.cc/65CRS3Nk/main-page.jpg)
#### Transcription
![2](https://i.postimg.cc/gcC8Nh1y/main-page-2.jpg)
#### Audio Modification
![3](https://i.postimg.cc/c1gfRhk4/main-page-3.jpg)
![4](https://i.postimg.cc/RqxSK4sf/main-page-4.jpg)
#### Speaker Diarization
![5](https://i.postimg.cc/XqCNrp84/main-page-5.jpg)
#### Downloads
![6](https://i.postimg.cc/W3t1P9qD/main-page-6.jpg)

### Live Transcription
![7](https://i.postimg.cc/Jhpzvw0V/real-time.jpg)

### Chatbot
![8](https://i.postimg.cc/W3KbpM05/chatbot.jpg)
