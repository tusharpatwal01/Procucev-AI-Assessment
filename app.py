import time
import streamlit as st
from database import init_db, create_transcription
from stt_service import LiveTranscriber
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Real-Time Speech-to-Text", layout="wide")
init_db()

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
)

tab_live, tab_history = st.tabs(["🎙️ Live Transcription", "📜 History"])

with tab_live:
    st.header("Real-Time Speech-to-Text")

    if "transcriber" not in st.session_state:
        st.session_state.transcriber = None
        st.session_state.live_text = ""

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Start Recording", disabled=not api_key):
            if not api_key:
                st.error("Please provide a Gemini API key.")
            else:
                tr = LiveTranscriber(api_key)
                tr.start()                     # starts background thread
                st.session_state.transcriber = tr
                st.session_state.live_text = ""
                st.success("Recording started…")

    with col2:
        if st.button("⏹️ Stop & Save"):
            tr = st.session_state.transcriber
            if tr:
                tr.stop()                     # stops background thread
                if st.session_state.live_text.strip():
                    tid = create_transcription(st.session_state.live_text.strip())
                    st.success(f"Transcription saved (ID: {tid})")
                else:
                    st.warning("No transcription to save.")
                st.session_state.transcriber = None
                st.session_state.live_text = ""

    placeholder = st.empty()

    # Poll the queue in a loop while the transcriber is running
    tr = st.session_state.transcriber
    if tr:
        while True:
            chunk = tr.get_transcript()       # non-blocking
            if chunk:
                st.session_state.live_text += chunk + " "
                placeholder.markdown(
                    f"**Live Transcript:** {st.session_state.live_text}"
                )
            if not tr._thread or not tr._thread.is_alive():
                break
            time.sleep(0.05)                 # small delay to avoid busy-waiting

    # Optional manual save
    if st.session_state.live_text.strip():
        if st.button("💾 Save Current Transcript"):
            tid = create_transcription(st.session_state.live_text.strip())
            st.success(f"Saved (ID: {tid})")
            st.session_state.live_text = ""