import os
import datetime
import pickle
import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

import util
# from test import test  # Assuming test is a function for anti-spoofing

st.title('Face Recognition App')

# Global variables
db_dir = './db'
log_path = './log.txt'
most_recent_capture_arr = None

# Ensure the database directory exists
if not os.path.exists(db_dir):
    os.mkdir(db_dir)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

    def get_frame(self):
        return self.frame

# Webcam video streamer
ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
processor = ctx.video_processor

def capture_frame():
    global most_recent_capture_arr
    if processor:
        most_recent_capture_arr = processor.get_frame()
        if most_recent_capture_arr is not None:
            st.image(most_recent_capture_arr, channels="BGR")

capture_frame()

def login():
    global most_recent_capture_arr

    if most_recent_capture_arr is None:
        util.msg_box('Error', 'No frame captured from webcam.')
        return

    # label = test(image=most_recent_capture_arr)  # Assuming test is an anti-spoofing check function

    label = 1  # Mocking successful anti-spoofing for demonstration

    if label == 1:
        name = util.recognize(most_recent_capture_arr, db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back !', f'Welcome, {name}.')
            with open(log_path, 'a') as f:
                f.write(f'{name},{datetime.datetime.now()},in\n')
    else:
        util.msg_box('Hey, you are a spoofer!', 'You are fake!')

def logout():
    global most_recent_capture_arr

    if most_recent_capture_arr is None:
        util.msg_box('Error', 'No frame captured from webcam.')
        return

    # label = test(image=most_recent_capture_arr)  # Assuming test is an anti-spoofing check function

    label = 1  # Mocking successful anti-spoofing for demonstration

    if label == 1:
        name = util.recognize(most_recent_capture_arr, db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Hasta la vista!', f'Goodbye, {name}.')
            with open(log_path, 'a') as f:
                f.write(f'{name},{datetime.datetime.now()},out\n')
    else:
        util.msg_box('Hey, you are a spoofer!', 'You are fake!')

def register_new_user():
    global most_recent_capture_arr

    if most_recent_capture_arr is None:
        util.msg_box('Error', 'No frame captured from webcam.')
        return

    name = st.text_input('Please, input username:')
    if st.button('Accept'):
        embeddings = face_recognition.face_encodings(most_recent_capture_arr)[0]

        with open(os.path.join(db_dir, f'{name}.pickle'), 'wb') as file:
            pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully!')

# Buttons for login, logout, and register
st.button('Login', on_click=login)
st.button('Logout', on_click=logout)
st.button('Register new user', on_click=register_new_user)
