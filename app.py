"""Streamlit demo:  streamlit run app.py"""

import os
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image

from detector import ViolenceDetector


@st.cache_resource
def load_detector():
    return ViolenceDetector()


st.title("Violence detection with CLIP")
st.caption("Zero-shot: no training on violence data. Upload an image or a short video.")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
upload = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"])

if upload is not None:
    detector = load_detector()
    if upload.type.startswith("image"):
        image = Image.open(upload).convert("RGB")
        p = detector.predict_image(image)
        st.image(image)
        st.metric("Violence probability", f"{p:.2f}", "violent" if p >= threshold else "non-violent", delta_color="off")
    else:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(upload.name)[1], delete=False) as f:
            f.write(upload.read())
        st.video(f.name)
        with st.spinner("Scoring frames..."):
            times, probs = detector.predict_video(f.name)
        os.remove(f.name)
        st.metric("Peak violence probability (smoothed)", f"{probs.max():.2f}",
                  "violent" if probs.max() >= threshold else "non-violent", delta_color="off")
        st.line_chart(pd.DataFrame({"violence probability": probs}, index=pd.Index(times.round(1), name="seconds")))
