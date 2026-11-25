import asyncio
import threading
import queue
import pyaudio
from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


class LiveTranscriber:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-transcribe-live"  # or your preferred live model
        self.config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[],  # auto language detection (Hindi + English)
            ),
        )

        # Thread‑safe queue used to pass transcripts back to Streamlit
        self.transcript_queue: queue.Queue[str] = queue.Queue()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- public API ----------
    def start(self):
        """Start the background thread that owns the asyncio loop and the live session."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_session, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and wait for it to finish."""
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def get_transcript(self) -> str | None:
        """Non‑blocking way for Streamlit to poll the next transcript chunk."""
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None

    # ---------- internal ----------
    def _run_session(self):
        """Entry point for the background thread – creates its own event loop."""
        asyncio.run(self._session_loop())

    async def _session_loop(self):
        """Open the Live API session with async with and keep it alive."""
        pya = pyaudio.PyAudio()
        mic = None
        try:
            mic = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )

            async with self.client.aio.live.connect(
                model=self.model, config=self.config
            ) as session:
                # Run sender and receiver concurrently
                await asyncio.gather(
                    self._send_audio(session, mic),
                    self._receive_transcripts(session),
                )
        finally:
            if mic:
                mic.stop_stream()
                mic.close()
            pya.terminate()

    async def _send_audio(self, session, mic):
        """Continuously read from the microphone and push PCM chunks."""
        while not self._stop_event.is_set():
            data = await asyncio.to_thread(
                mic.read, CHUNK_SIZE, exception_on_overflow=False
            )
            await session.send_realtime_input(
                audio={"data": data, "mime_type": "audio/pcm"}
            )

    async def _receive_transcripts(self, session):
        """Listen for transcription events and put them into the thread‑safe queue."""
        async for response in session.receive():
            if (
                response.server_content
                and response.server_content.input_transcription
                and response.server_content.input_transcription.text
            ):
                self.transcript_queue.put(
                    response.server_content.input_transcription.text
                )