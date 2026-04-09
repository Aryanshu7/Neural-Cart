import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pyttsx3
import speech_recognition as sr
import threading

# -------------------- SETTINGS --------------------
st.set_page_config(page_title="Smart AI Cart", layout="centered")
st.title("🛒 Smart Retail AI Cart")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()

# -------------------- LOAD LABELS --------------------
def load_labels():
    labels = []
    with open("labels.txt", "r") as f:
        for line in f.readlines():
            label = line.strip()
            if " " in label:
                label = label.split(" ", 1)[1]
            labels.append(label.lower().replace(" ", ""))
    return labels

class_names = load_labels()

# -------------------- PRICE LIST --------------------
prices = {
    "maggie": 14,
    "soap": 25,
    "lays": 20,
    "oreo": 30,
    "cocacola": 40
}

# -------------------- VOICE OUTPUT (NON-BLOCKING) --------------------
def speak(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run).start()

# -------------------- VOICE INPUT --------------------
def listen_command():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)

        command = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {command}")
        return command.lower()

    except:
        st.warning("Voice not recognized")
        return ""

# -------------------- SESSION STATE --------------------
if "cart" not in st.session_state:
    st.session_state.cart = {}

# -------------------- CAMERA --------------------
st.subheader("📷 Scan Product")
img = st.camera_input("Take a picture")

detected_label = None
confidence = 0

if img is not None:
    image = Image.open(img).convert("RGB")
    image = image.resize((224, 224))

    # ✅ CORRECT NORMALIZATION
    image = (np.array(image).astype(np.float32) / 127.5) - 1
    image = np.reshape(image, (1, 224, 224, 3))

    prediction = model.predict(image, verbose=0)[0]

    index = np.argmax(prediction)
    confidence = float(prediction[index])

    # ✅ NOTHING DETECTED LOGIC
    if confidence > 0.90:
        detected_label = class_names[index]
    else:
        detected_label = None

    # ---------------- DISPLAY ----------------
    if detected_label:
        st.write(f"🔍 Detected: **{detected_label}** ({confidence:.2f})")
        st.progress(int(confidence * 100))
    else:
        st.error("❌ Nothing detected")

# -------------------- ADD BUTTON --------------------
if detected_label:
    if st.button("➕ Add to Cart"):

        if detected_label in prices:
            st.session_state.cart[detected_label] = st.session_state.cart.get(detected_label, 0) + 1

            total = sum(prices[i]*q for i,q in st.session_state.cart.items())

            st.success(f"{detected_label} added")
            speak(f"{detected_label} added. Total is {total} rupees")
        else:
            st.error("Item not in price list")

# -------------------- VOICE COMMAND --------------------
st.subheader("🎤 Voice Assistant")

if st.button("🎤 Start Voice Command"):
    command = listen_command()

    if command:

        if any(word in command for word in ["clear", "reset"]):
            st.session_state.cart = {}
            speak("Cart cleared")

        elif "total" in command or "amount" in command:
            total = sum(prices[i]*q for i,q in st.session_state.cart.items())
            speak(f"Your total is {total} rupees")

        elif "remove" in command:
            found = False
            for item in list(st.session_state.cart.keys()):
                if item in command:
                    del st.session_state.cart[item]
                    speak(f"{item} removed")
                    found = True
                    break
            if not found:
                speak("Item not found")

        elif "add" in command:
            found = False
            for item in prices:
                if item in command:
                    st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
                    speak(f"{item} added")
                    found = True
                    break
            if not found:
                speak("Item not recognized")

        else:
            speak("Command not understood")

# -------------------- CART --------------------
st.subheader("🛍 Cart")

total = 0

if not st.session_state.cart:
    st.write("Cart is empty")
else:
    for item, qty in st.session_state.cart.items():
        item_total = prices[item] * qty

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"{item} x{qty} = ₹{item_total}")

        with col2:
            if st.button("❌", key=item):
                del st.session_state.cart[item]
                st.rerun()

        total += item_total

# -------------------- TOTAL --------------------
st.subheader(f"💰 Total: ₹{total}")

# -------------------- BILL --------------------
if st.button("🧾 Generate Bill"):
    st.subheader("Final Bill")

    for item, qty in st.session_state.cart.items():
        st.write(f"{item} x{qty} = ₹{prices[item]*qty}")

    st.write("----------")
    st.write(f"Total: ₹{total}")

    speak(f"Your final bill is {total} rupees")

# -------------------- CLEAR --------------------
if st.button("🗑 Clear Cart"):
    st.session_state.cart = {}
    st.rerun()