import os
import io
import time
import hashlib
import logging
import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder

from logging_config import setup_logging

setup_logging()
log = logging.getLogger("app")

from database import (
    init_db,
    create_transcription,
    get_all_transcriptions,
    update_transcription,
    delete_transcription,
)
from stt_service import LiveTranscriber

load_dotenv()
log.info("app: dotenv loaded, GEMINI_API_KEY set=%s", bool(os.getenv("GEMINI_API_KEY")))

st.set_page_config(page_title="Real-Time Speech-to-Text", layout="wide")
init_db()
log.info("app: database initialized")

st.sidebar.header("Configuration")
env_key = os.getenv("GEMINI_API_KEY", "")
if env_key:
    st.sidebar.success("✅ GEMINI_API_KEY loaded from .env")
    api_key = env_key
    log.info("app: using GEMINI_API_KEY from .env (len=%d)", len(env_key))
else:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    log.info("app: API key from sidebar, len=%d", len(api_key or ""))

for key, default in [
    ("live_text", ""),
    ("transcriber", None),
    ("last_audio_hash", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

tab_live, tab_history = st.tabs(["🎙️ Transcribe", "📜 History"])

with tab_live:
    st.header("Real-Time Speech-to-Text")

    if not api_key:
        st.warning("Please set `GEMINI_API_KEY` in `.env` or use the sidebar.")
    else:
        audio_segment = audiorecorder(
            start_prompt="▶️ Start recording",
            stop_prompt="⏹️ Stop recording",
            pause_prompt="",
            custom_style={"color": "#1f77b4"},
            show_visualizer=True,
            key="voice_recorder",
        )

        if audio_segment is not None:
            log.info(
                "app: audiorecorder returned AudioSegment, len_ms=%s",
                len(audio_segment) if hasattr(audio_segment, "__len__") else "?",
            )
        else:
            log.debug("app: audiorecorder returned None (no recording)")

        if audio_segment and len(audio_segment) > 0:
            log.info("app: exporting AudioSegment -> WAV bytes")
            buf = io.BytesIO()
            audio_segment.export(buf, format="wav")
            audio_bytes = buf.getvalue()
            log.info("app: exported %d WAV bytes", len(audio_bytes))

            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            log.info(
                "app: audio hash=%s (prev=%s)",
                audio_hash,
                st.session_state.last_audio_hash,
            )

            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.session_state.live_text = ""
                log.info("app: new recording detected — starting transcriber")
                tr = LiveTranscriber(api_key)
                tr.start(audio_bytes)
                st.session_state.transcriber = tr
            else:
                log.info("app: same recording as before, skipping start")

        tr = st.session_state.transcriber
        if tr is not None:
            log.info(
                "app: transcriber exists — entering poll loop "
                "(done=%s, queue_size=%d, error=%r)",
                tr.done.is_set(),
                tr.transcript_queue.qsize(),
                tr.error,
            )
            placeholder = st.empty()
            poll_count = 0
            with st.spinner("Transcribing…"):
                while not tr.done.is_set() or not tr.transcript_queue.empty():
                    chunk = tr.get_chunk()
                    if chunk:
                        log.info("app: got chunk %r", chunk)
                        st.session_state.live_text += chunk + " "
                        placeholder.markdown(
                            f"**Transcript:** {st.session_state.live_text}"
                        )
                    poll_count += 1
                    if poll_count % 50 == 0:
                        log.debug(
                            "app: poll #%d | done=%s queue=%d text_len=%d",
                            poll_count,
                            tr.done.is_set(),
                            tr.transcript_queue.qsize(),
                            len(st.session_state.live_text),
                        )
                    time.sleep(0.1)
            log.info(
                "app: poll loop exited after %d iterations | final_text=%r "
                "| error=%r",
                poll_count,
                st.session_state.live_text,
                tr.error,
            )

            if tr.error:
                st.error(f"Transcription error: {tr.error}")
            elif st.session_state.live_text.strip():
                placeholder.markdown(f"**Transcript:** {st.session_state.live_text}")
            else:
                log.warning("app: transcriber finished with NO text and no error")
                st.warning(
                    "Transcriber finished but returned no text. "
                    "Check the console logs for message details. "
                    "Try recording a longer, clearer sentence (2–4 seconds)."
                )

        if st.session_state.live_text.strip():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Transcription", use_container_width=True):
                    log.info("app: saving transcription")
                    tid = create_transcription(st.session_state.live_text.strip())
                    st.success(f"Saved (ID: {tid})")
                    st.session_state.live_text = ""
                    st.session_state.transcriber = None
                    st.session_state.last_audio_hash = None
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    log.info("app: clear pressed")
                    st.session_state.live_text = ""
                    st.session_state.transcriber = None
                    st.session_state.last_audio_hash = None
                    st.rerun()

# ---- history tab unchanged ----
with tab_history:
    st.header("Transcription History")
    rows = get_all_transcriptions()
    if not rows:
        st.info("No transcriptions yet.")
    else:
        for row in rows:
            with st.expander(f"#{row['id']} — {row['created_at'][:19]}"):
                col_a, col_b, col_c = st.columns([4, 1, 1])
                with col_a:
                    new_text = st.text_area(
                        "Transcript", value=row["text"], key=f"text_{row['id']}"
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
