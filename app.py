import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import speech_recognition as sr
import pyttsx3

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Neural Cart", layout="centered")

# -------------------- MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()

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

suggestions = {
    "maggie": ["cocacola", "lays"],
    "oreo": ["cocacola"],
    "lays": ["cocacola"]
}

# -------------------- SESSION --------------------
if "cart" not in st.session_state:
    st.session_state.cart = {}

if "last_detect_time" not in st.session_state:
    st.session_state.last_detect_time = 0

# -------------------- VOICE --------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio).lower()
        return command
    except:
        return ""

# -------------------- HEADER --------------------
st.title("🧠 Neural Cart")

# -------------------- CAMERA --------------------
st.subheader("📷 Scan Product")
img = st.camera_input("Capture")

detected_label = None
confidence = 0

if img is not None:
    image = Image.open(img).convert("RGB")
    image = image.resize((224, 224))

    image = (np.array(image).astype(np.float32) / 127.5) - 1
    image = np.reshape(image, (1, 224, 224, 3))

    prediction = model.predict(image, verbose=0)[0]

    top2 = np.argsort(prediction)[-2:]
    best, second = top2[-1], top2[-2]

    detected_label = class_names[best]
    confidence = prediction[best]
    gap = prediction[best] - prediction[second]

    # ---------------- SMART FILTER ----------------
    if confidence < 0.85 or gap < 0.15:
        st.warning("❌ Nothing confidently detected")
        detected_label = None
    else:
        st.success(f"Detected: {detected_label} ({confidence:.2f})")

# ---------------- MANUAL FIX ----------------
if detected_label:
    choice = st.selectbox("Correct item if wrong:", class_names, index=class_names.index(detected_label))

    if st.button("➕ Add Item"):
        now = time.time()
        if now - st.session_state.last_detect_time > 3:
            st.session_state.cart[choice] = st.session_state.cart.get(choice, 0) + 1
            st.session_state.last_detect_time = now
            speak(f"{choice} added")
        else:
            st.warning("Wait before scanning again")

# ---------------- VOICE CONTROL ----------------
st.subheader("🎤 Voice Commands")

if st.button("Start Voice"):
    cmd = listen()

    if "add" in cmd:
        for item in prices:
            if item in cmd:
                st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
                speak(f"{item} added")

    elif "remove" in cmd:
        for item in list(st.session_state.cart.keys()):
            if item in cmd:
                del st.session_state.cart[item]
                speak(f"{item} removed")

# ---------------- CART --------------------
st.subheader("🛒 Cart")

total = 0

for item, qty in list(st.session_state.cart.items()):
    col1, col2, col3, col4 = st.columns([3,1,1,1])

    col1.write(f"{item.upper()} - ₹{prices[item]}")

    if col2.button("➕", key=f"add_{item}"):
        st.session_state.cart[item] += 1

    if col3.button("➖", key=f"sub_{item}"):
        if qty > 1:
            st.session_state.cart[item] -= 1
        else:
            del st.session_state.cart[item]

    if col4.button("❌", key=f"del_{item}"):
        del st.session_state.cart[item]

    total += prices[item] * qty

st.metric("Total", f"₹{total}")

# ---------------- SMART SUGGESTIONS ----------------
st.subheader("🧠 Smart Suggestions")

recommended = set()

for item in st.session_state.cart:
    if item in suggestions:
        recommended.update(suggestions[item])

for r in recommended:
    if r not in st.session_state.cart:
        if st.button(f"Add {r}"):
            st.session_state.cart[r] = 1

# ---------------- BILL --------------------
if st.button("🧾 Generate Bill"):
    st.subheader("Final Bill")
    for item, qty in st.session_state.cart.items():
        st.write(f"{item} x{qty} = ₹{prices[item]*qty}")
    st.write(f"**Total: ₹{total}**")

# ---------------- CLEAR --------------------
if st.button("🗑 Clear Cart"):
    st.session_state.cart = {}
    st.rerun()
