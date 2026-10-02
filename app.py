"""
Streamlit UI only — no training logic here (see train.py) and no
data-handling beyond what's needed to process one uploaded image:
everything happens in memory, nothing uploaded is written to disk or
sent to any third-party API. See SPEC.md -> "Data security / privacy
constraints".
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from data import IMAGENET_MEAN, IMAGENET_STD
from model import EMOTION_LABELS, build_model

MAX_UPLOAD_MB = 10
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "resnet_best.pt")
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "samples", "sample_face.jpg")

EMOTION_COLORS = {
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Angry": (0, 0, 255),
    "Surprise": (0, 255, 255),
    "Fear": (128, 0, 128),
    "Disgust": (128, 128, 0),
    "Neutral": (128, 128, 128),
}

st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="wide")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT_PATH):
        return None, device
    model = build_model("resnet", pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """BGR face crop -> normalized 3x224x224 tensor, matching the
    transform used at training time for the ResNet model
    (see data.get_transform('resnet', train=False))."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)  # duplicate channel, matches training transform
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def predict_emotion(model, device, face_bgr: np.ndarray):
    tensor = preprocess_face(face_bgr).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    predicted_idx = int(probs.argmax())
    return EMOTION_LABELS[predicted_idx], float(probs[predicted_idx]) * 100, probs.cpu().numpy() * 100


def process_image(pil_image: Image.Image, model, device, face_cascade):
    """Runs face detection + per-face emotion prediction, entirely in
    memory. Returns an annotated RGB array and a list of per-face results."""
    img_array = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    results = []
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = bgr[y : y + h, x : x + w]
        emotion, confidence, all_probs = predict_emotion(model, device, face_roi)
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 3)
        cv2.putText(bgr, f"{emotion} {confidence:.0f}%", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        results.append({
            "face_number": i + 1,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": all_probs,
        })

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), results


def render_sidebar():
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown(
        """
### How to use
1. Upload a JPG/PNG photo, or use the sample below
2. Click **Analyze**
3. See the detected emotion per face

### Model
- **Architecture:** ResNet18 (ImageNet-pretrained), fine-tuned on FER-2013
- **Face detection:** OpenCV Haar Cascade
- **Classes:** 7 emotions
- See the [project README](https://github.com/slastrzelec/emotion-recognition-cv)
  for measured accuracy and how it compares to the from-scratch baseline.
"""
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**Sławomir Strzelec** — AI/ML Engineer
[LinkedIn](https://www.linkedin.com/in/s%C5%82awomir-strzelec/) ·
[GitHub](https://github.com/slastrzelec) ·
[Repository](https://github.com/slastrzelec/emotion-recognition-cv)
"""
    )


def main():
    st.title("😊 Emotion Recognition")
    st.caption("Face detection (OpenCV) + emotion classification (fine-tuned ResNet18) — local inference, nothing uploaded is stored.")

    render_sidebar()

    model, device = load_model()
    face_cascade = load_face_cascade()

    if model is None:
        st.error(
            "No trained model checkpoint found. This app expects "
            "`checkpoints/resnet_best.pt`, produced by `train.py` — see the README."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Image")
        uploaded = st.file_uploader("Upload a photo", type=ALLOWED_TYPES)
        if uploaded is not None:
            st.session_state.use_sample = False
        if uploaded is None and os.path.exists(SAMPLE_IMAGE_PATH):
            if st.button("Use the sample photo"):
                st.session_state.use_sample = True

        using_sample = uploaded is None and st.session_state.get("use_sample", False)

        image = None
        if using_sample:
            st.caption("Using the bundled sample photo.")
            image = Image.open(SAMPLE_IMAGE_PATH)
        elif uploaded is not None:
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File too large (max {MAX_UPLOAD_MB} MB).")
            else:
                try:
                    image = Image.open(uploaded)
                except Exception:
                    st.error("Could not read this file as an image. Please upload a valid JPG or PNG.")

        if image is not None:
            st.image(image, use_container_width=True)
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    annotated, results = process_image(image, model, device, face_cascade)
                    st.session_state["annotated"] = annotated
                    st.session_state["results"] = results
        else:
            st.info("Upload a photo or use the sample above.")

    with col2:
        st.subheader("📊 Results")
        results = st.session_state.get("results")
        if results:
            st.image(st.session_state["annotated"], use_container_width=True)
            for r in results:
                st.markdown(f"**Face {r['face_number']}:** {r['emotion']} ({r['confidence']:.0f}%)")
                probs_dict = {EMOTION_LABELS[i]: float(r["probabilities"][i]) for i in range(len(EMOTION_LABELS))}
                st.bar_chart(probs_dict)
        elif results is not None:
            st.warning("No faces detected in this image.")
        else:
            st.info("Analyze an image to see results here.")


if __name__ == "__main__":
    main()
