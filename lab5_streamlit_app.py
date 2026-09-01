"""
RGIPT - CS361 AIMLDL Application
Lab Assignment 5 : Deploying the Lab-4 CNN as a web app (Streamlit)

RUN WITH:
    streamlit run lab5_streamlit_app.py

WHAT IT DOES
------------
- Loads outputs_lab4/best_model.keras (the winner from Lab 4) and
  outputs_lab4/class_names.json (so labels line up correctly).
- Lets the user upload a clothing image (jpg/png).
- Preprocesses it exactly the way training data was preprocessed
  (resize to 128x128, rescale to [0,1]).
- Shows the predicted class, confidence, and the top-3 predictions
  as a bar chart.

If you'd rather deploy with Flask or Django, see lab5_flask_app.py in the
same folder for an equivalent Flask version.
"""

import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL_PATH = "outputs_lab4/best_model.keras"
CLASS_NAMES_PATH = "outputs_lab4/class_names.json"
IMG_SIZE = 128


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)


def main():
    st.set_page_config(page_title="Clothing Classifier", page_icon="\U0001F455", layout="centered")
    st.title("Clothing Image Classifier")
    st.caption("Lab Assignment 5 -- CNN trained in Lab 4, deployed with Streamlit")

    try:
        model, class_names = load_model_and_classes()
    except Exception as e:
        st.error(
            "Could not load the trained model. Run lab4_cnn_experiments.py first so "
            f"that {MODEL_PATH} and {CLASS_NAMES_PATH} exist.\n\nDetails: {e}"
        )
        return

    uploaded = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running inference..."):
            x = preprocess(image)
            probs = model.predict(x, verbose=0)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        pred_idx = top3_idx[0]

        with col2:
            st.subheader("Prediction")
            st.markdown(f"### {class_names[pred_idx]}")
            st.metric("Confidence", f"{probs[pred_idx]*100:.2f}%")

        st.subheader("Top-3 predictions")
        top3_labels = [class_names[i] for i in top3_idx]
        top3_scores = [float(probs[i]) for i in top3_idx]
        st.bar_chart({"confidence": dict(zip(top3_labels, top3_scores))})

        with st.expander("Raw probability vector"):
            st.json({class_names[i]: float(probs[i]) for i in range(len(class_names))})
    else:
        st.info("Upload an image to get a prediction.")


if __name__ == "__main__":
    main()
