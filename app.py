import os
import io
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder   # from streamlit-audiorecorder

from database import (
    init_db,
    create_transcription,
    get_all_transcriptions,
    update_transcription,
    delete_transcription,
)
from stt_service import LiveTranscriber

# ------------------------------------------------------------------ setup
load_dotenv()
st.set_page_config(page_title="Real-Time Speech-to-Text", layout="wide")
init_db()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Configuration")

env_key = os.getenv("GEMINI_API_KEY", "")
if env_key:
    st.sidebar.success("✅ GEMINI_API_KEY loaded from .env")
    api_key = env_key
else:
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Set GEMINI_API_KEY in .env (local) or Streamlit Secrets (cloud).",
    )

# ------------------------------------------------------------------ session state
for key, default in [
    ("live_text", ""),
    ("transcriber", None),
    ("last_audio_hash", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------ tabs
tab_live, tab_history = st.tabs(["🎙️ Transcribe", "📜 History"])

# ================================================================== LIVE TAB
with tab_live:
    st.header("Real-Time Speech-to-Text")
    st.caption(
        "Speak a sentence in Hindi, English, or a mix of both. "
        "Click the mic, speak, then click again to stop."
    )

    if not api_key:
        st.warning("Please set `GEMINI_API_KEY` in `.env` or use the sidebar.")
    else:
        # ✅ CORRECT parameters for audiorecorder (streamlit-audiorecorder)
        audio_segment = audiorecorder(
            start_prompt="▶️ Start recording",
            stop_prompt="⏹️ Stop recording",
            pause_prompt="",                # leave empty to hide pause button
            custom_style={"color": "#1f77b4"},
            show_visualizer=True,
            key="voice_recorder",
        )

        # audiorecorder returns a pydub.AudioSegment, not raw bytes.
        # Convert to WAV bytes only when a new recording exists.
        if audio_segment and len(audio_segment) > 0:
            # Export AudioSegment to WAV bytes
            buf = io.BytesIO()
            audio_segment.export(buf, format="wav")
            audio_bytes = buf.getvalue()

            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.session_state.live_text = ""
                tr = LiveTranscriber(api_key)
                tr.start(audio_bytes)          # bytes passed to backend
                st.session_state.transcriber = tr

        # Progressive display while transcription runs
        tr = st.session_state.transcriber
        if tr:
            placeholder = st.empty()
            with st.spinner("Transcribing…"):
                while not tr.done.is_set() or not tr.transcript_queue.empty():
                    chunk = tr.get_chunk()
                    if chunk:
                        st.session_state.live_text += chunk + " "
                        placeholder.markdown(
                            f"**Transcript:** {st.session_state.live_text}"
                        )
                    time.sleep(0.1)

            if tr.error:
                st.error(f"Transcription error: {tr.error}")
            elif st.session_state.live_text.strip():
                placeholder.markdown(
                    f"**Transcript:** {st.session_state.live_text}"
                )

        # Save / clear controls
        if st.session_state.live_text.strip():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Transcription", use_container_width=True):
                    tid = create_transcription(
                        st.session_state.live_text.strip()
                    )
                    st.success(f"Saved (ID: {tid})")
                    st.session_state.live_text = ""
                    st.session_state.transcriber = None
                    st.session_state.last_audio_hash = None
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.live_text = ""
                    st.session_state.transcriber = None
                    st.session_state.last_audio_hash = None
                    st.rerun()

# ================================================================== HISTORY TAB
with tab_history:
    st.header("Transcription History")

    rows = get_all_transcriptions()
    if not rows:
        st.info("No transcriptions yet. Record something in the Transcribe tab!")
    else:
        for row in rows:
            with st.expander(f"#{row['id']} — {row['created_at'][:19]}"):
                col_a, col_b, col_c = st.columns([4, 1, 1])
                with col_a:
                    new_text = st.text_area(
                        "Transcript",
                        value=row["text"],
                        key=f"text_{row['id']}",
                    )
                with col_b:
                    if st.button("Update", key=f"upd_{row['id']}"):
                        update_transcription(row["id"], new_text)
                        st.success("Updated!")
                        st.rerun()
                with col_c:
                    if st.button("Delete", key=f"del_{row['id']}"):
                        delete_transcription(row["id"])
                        st.success("Deleted!")
                        st.rerun()