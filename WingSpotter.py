import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, Model
import base64
from io import BytesIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="WingSpotter",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background-color: #EFE8E3;
    font-family: Inter, sans-serif;
}

/* NAVBAR */
.navbar {
    position: fixed;
    top: 0;
    width: 100%;
    background: #EFE8E3;
    padding: 16px 70px;
    border-bottom: 1px solid #d6ccc7;
    z-index: 1000;
}
.nav-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.brand {
    font-size: 22px;
    font-weight: 700;
    color: #2f5755;
    display: flex;
    gap: 10px;
}
.logo {
    background: #2f5755;
    color: white;
    padding: 6px 10px;
    border-radius: 8px;
}
.nav-links span {
    margin: 0 18px;
    font-weight: 500;
    color: #4b2e2b;
}
.nav-btn {
    background: #2f5755;
    color: white;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 600;
}

/* HERO — FULL WIDTH */
.hero {
    margin-top: -20px;
    padding: 60px 120px;
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    border: 2px solid #7fa7a2;
    color: #2f5755;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 72px;
    color: #4b2e2b;
    line-height: 1.1;
    max-width: 1000px;
}
.hero h1 span {
    color: #2f5755;
}
.hero p {
    font-size: 20px;
    color: #6f5a56;
    max-width: 800px;
    margin-top: 16px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] label{
    background: transparent;
    border: none;
    color: black !important;
    font-weight: 600;
}
[data-testid="stFileUploader"] section {
    background: #2f5755;
    color: white;
    padding: 14px;
    border-radius: 14px;
}
[data-testid="stFileUploader"] button {
    background: white;
    color: #2f5755;
    border-radius: 20px;
    font-weight: 600;
}

/* RESULT GRID */
.result-grid {
    display: grid;
    grid-template-columns: 40% 40%;
    justify-content: center;
    gap: 4%;
    padding: 60px 120px;
    width: 100%;
}

/* IMAGE BOX */
.image-box {
    width: 100%;
    height: 460px;
    background: black;
    border-radius: 20px;
    overflow: hidden;
}
.image-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* PREDICTION BOX */
.prediction-card {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    width: 100%;
    height: 460px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.prediction-card h2 {
    color: black;
    margin-bottom: 20px;
}
.prediction-card h1 {
    color: #2f5755;
    margin-bottom: 20px;
    font-size: 42px;
}
.prediction-card p {
    color: #6f5a56;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# NAVBAR
# =========================
st.markdown("""
<div class="navbar">
  <div class="nav-inner">
    <div class="brand">
      <div class="logo">🪶</div> WingSpotter
    </div>
    <div class="nav-links">
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <div class="badge">● Powered by Advanced AI</div>
    <h1>Identify Birds<br><span>Instantly</span> with AI</h1>
    <p>
        Upload a bird photo and our AI instantly identifies the species.
        Built for birdwatchers, researchers, and nature lovers.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD (BELOW HERO)
# =========================
st.markdown("<div style='padding: 0 120px 40px;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a bird image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
base = tf.keras.applications.efficientnet.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="max"
)
base.trainable = False

x = layers.Dense(128, activation="relu")(base.output)
x = layers.Dropout(0.45)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.45)(x)
outputs = layers.Dense(525, activation="softmax")(x)

model = Model(base.input, outputs)
model.load_weights("model_weights.h5")
labels = np.load("labels.npy", allow_pickle=True).item()

# =========================
# RESULT SECTION
# =========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    arr = np.expand_dims(np.array(img.resize((224, 224))), axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    confidence = preds[0][idx] * 100

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
    f"""
    <div class="result-grid">

      <div class="image-box">
        <img src="data:image/png;base64,{img_base64}" />
      </div>

      <div class="prediction-card">
        <h2>Prediction</h2>
        <h1>{labels[idx]}</h1>
        <p><b>Confidence:</b> {confidence:.2f}%</p>
      </div>

    </div>
    """,
    unsafe_allow_html=True
)


