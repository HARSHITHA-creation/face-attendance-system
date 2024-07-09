import os
import pickle
import streamlit as st
import face_recognition
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def get_button(window, text, color, command, fg='white'):
    if st.button(text):
        command()

def get_img_label(window):
    return st.empty()

def get_text_label(window, text):
    st.text(text)

def get_entry_text(window):
    return st.text_input("Enter text:")

def msg_box(title, description):
    st.info(description)

def recognize(img, db_path):
    # it is assumed there will be at most 1 match in the db
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted(os.listdir(db_path))

    match = False
    j = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])

        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)

        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        j += 1

    if match:
        return db_dir[j - 1][:-7]
    else:
        return 'unknown_person'
