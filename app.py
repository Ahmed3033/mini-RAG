import streamlit as st
from rag import ask_rag

st.set_page_config(page_title="Academic RAG Bot", page_icon="🎓")

st.title("🎓 Academic Assistant Bot")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask your question...")

if user_input:

    answer, sources = ask_rag(user_input)

    st.session_state.chat.append((user_input, answer, sources))

# =========================
# DISPLAY CHAT
# =========================

for q, a, s in st.session_state.chat:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

    if s:
        with st.expander("📚 Sources"):
            for doc in s:
                st.write(doc.page_content[:400])
