import streamlit as st
import random

from rag import ask_rag

# =====================
# UI
# =====================
st.set_page_config(
    page_title="Academic Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Academic AI Assistant")

# =====================
# GREETING
# =====================
def is_greeting(text):
    text = text.lower()
    return any(x in text for x in [
        "hi", "hello", "hey",
        "سلام", "ازيك", "اهلا", "مرحبا"
    ])

def greet():
    return random.choice([
        "Hello 👋 How can I help you?",
        "Hi 😊 Ask your academic question.",
        "Hey 🎓 I'm ready to help!"
    ])

# =====================
# SESSION
# =====================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =====================
# INPUT
# =====================
msg = st.chat_input("Ask your question...")

if msg:

    st.session_state.chat.append(("user", msg))

    if is_greeting(msg):
        reply = greet()
    else:
        reply = ask_rag(msg)

    st.session_state.chat.append(("bot", reply))

# =====================
# CHAT UI
# =====================
for role, text in st.session_state.chat:

    if role == "user":
        with st.chat_message("user"):
            st.markdown(text)

    else:
        with st.chat_message("assistant"):
            st.markdown(text)

# =====================
# DEBUG
# =====================
st.sidebar.title("Debug")

if st.session_state.chat:
    st.sidebar.write(st.session_state.chat[-1][1])