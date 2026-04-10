import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Neural Cart", layout="centered")

# -------------------- UI STYLE --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    text-align: center;
}

.stButton>button {
    background: linear-gradient(45deg, #38bdf8, #0ea5e9);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.card {
    background: #1e293b;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<h1>🧠 Neural Cart</h1>
<p style='text-align:center; color:gray;'>AI-powered smart shopping experience</p>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()

# -------------------- LABELS --------------------
def load_labels():
    labels = []
    with open("labels.txt", "r") as f:
        for line in f.readlines():
            label = line.strip().split(" ",1)[-1]
            labels.append(label.lower().replace(" ",""))
    return labels

class_names = load_labels()

# -------------------- DATA --------------------
prices = {
    "maggie": 14,
    "soap": 25,
    "lays": 20,
    "oreo": 30,
    "cocacola": 40
}

images = {
    "maggie": "https://upload.wikimedia.org/wikipedia/commons/7/7b/Maggi_noodles.jpg",
    "soap": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Bar_of_soap.jpg",
    "lays": "https://upload.wikimedia.org/wikipedia/commons/6/69/Lay%27s_classic.jpg",
    "oreo": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Oreo-Two-Cookies.jpg",
    "cocacola": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Coca-Cola_can.jpg"
}

suggestions = {
    "maggie": ["cocacola", "lays"],
    "oreo": ["cocacola"],
    "lays": ["cocacola"]
}

# -------------------- SESSION --------------------
if "cart" not in st.session_state:
    st.session_state.cart = {}

if "last_added" not in st.session_state:
    st.session_state.last_added = None

# -------------------- CAMERA --------------------
st.markdown("### 📷 Scan Product")
img = st.camera_input("Take a picture")

if img is not None:
    image = Image.open(img).convert("RGB")
    image = image.resize((224, 224))

    image = (np.array(image).astype(np.float32) / 127.5) - 1
    image = np.reshape(image, (1, 224, 224, 3))

    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    label = class_names[index]
    confidence = float(prediction[0][index])

    # -------------------- DETECTION UI --------------------
    st.markdown(f"""
    <div class="card">
    <h3>🧠 Detected Item</h3>
    <h2>{label.upper()}</h2>
    <p>Confidence: {confidence:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if label in images:
            st.image(images[label], use_container_width=True)

    with col2:
        st.markdown(f"""
        **Item:** {label.upper()}  
        **Confidence:** {confidence:.2f}
        """)

    # -------------------- AUTO ADD --------------------
    if confidence > 0.90:
        if label in prices and st.session_state.last_added != label:
            st.session_state.cart[label] = st.session_state.cart.get(label, 0) + 1
            st.session_state.last_added = label
            st.success(f"🛒 {label} added automatically")

    else:
        st.warning("⚠️ Low confidence. Try again.")

# -------------------- DIVIDER --------------------
st.markdown("---")

# -------------------- CART --------------------
st.markdown("### 🛒 Your Cart")

total = 0

if not st.session_state.cart:
    st.info("Cart is empty")
else:
    for item, qty in st.session_state.cart.items():
        item_total = prices[item] * qty

        st.markdown(f"""
        <div class="card">
        <b>{item.upper()}</b><br>
        Quantity: {qty}<br>
        Total: ₹{item_total}
        </div>
        """, unsafe_allow_html=True)

        total += item_total

# -------------------- TOTAL --------------------
st.metric("💰 Total Bill", f"₹{total}")

# -------------------- SUGGESTIONS --------------------
st.markdown("### 🧠 Smart Suggestions")

for item in st.session_state.cart:
    if item in suggestions:
        for sug in suggestions[item]:
            if sug not in st.session_state.cart:
                st.write(f"👉 People also buy: {sug}")

# -------------------- BILL --------------------
if st.button("🧾 Generate Bill"):
    st.markdown("### 🧾 Final Bill")
    for item, qty in st.session_state.cart.items():
        st.write(f"{item.upper()} x{qty} = ₹{prices[item]*qty}")
    st.write("------")
    st.write(f"**Total: ₹{total}**")

# -------------------- CLEAR --------------------
if st.button("🗑 Clear Cart"):
    st.session_state.cart = {}
    st.session_state.last_added = None
    st.rerun()
