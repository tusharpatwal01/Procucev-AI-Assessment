"""
Speech-to-Text service using the Gemini Live API.

Instead of reading from a server-side microphone (which does not exist on
Streamlit Cloud), this service accepts pre-recorded WAV bytes produced by the
browser recorder widget, converts them to the 16 kHz / mono / 16-bit PCM
format required by the Gemini Live API, streams them to the model, and pushes
incremental transcription chunks into a thread-safe queue for the UI.
"""

import asyncio
import io
import threading
import queue
import wave
import audioop

from google import genai
from google.genai import types

TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLE_WIDTH = 2  # 16-bit
TARGET_CHANNELS = 1

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"


def wav_to_pcm_16k_mono(wav_bytes: bytes) -> bytes:
    """Convert arbitrary WAV bytes to 16 kHz / mono / 16-bit PCM."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        src_rate = wf.getframerate()
        src_channels = wf.getnchannels()
        src_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if src_channels > 1:
        frames = audioop.tomono(frames, src_width, 0.5, 0.5)

    if src_width != TARGET_SAMPLE_WIDTH:
        frames = audioop.lin2lin(frames, src_width, TARGET_SAMPLE_WIDTH)

    if src_rate != TARGET_SAMPLE_RATE:
        frames, _ = audioop.ratecv(
            frames, TARGET_SAMPLE_WIDTH, 1, src_rate, TARGET_SAMPLE_RATE, None
        )

    return frames


class LiveTranscriber:
    """
    Transcribes a pre-recorded WAV blob via Gemini Live API.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(api_version="v1beta")
        )
        self.model = model
        self.config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[],  # auto-detect (Hindi + English code-mixing)
            ),
        )
        self.transcript_queue: "queue.Queue[str]" = queue.Queue()
        self.done = threading.Event()
        self.error: str | None = None
        self._thread: "threading.Thread | None" = None

    # ------------------------------------------------------------------ public
    def start(self, wav_bytes: bytes):
        if self._thread and self._thread.is_alive():
            return
        self.done.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run, args=(wav_bytes,), daemon=True
        )
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_chunk(self) -> str | None:
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None

    # ----------------------------------------------------------------- internal
    def _run(self, wav_bytes: bytes):
        try:
            asyncio.run(self._transcribe(wav_bytes))
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
        finally:
            self.done.set()

    async def _transcribe(self, wav_bytes: bytes):
        pcm = wav_to_pcm_16k_mono(wav_bytes)

        async with self.client.aio.live.connect(
            model=self.model, config=self.config
        ) as session:

            async def send_audio():
                chunk_size = 3200  # 100 ms @ 16 kHz * 16-bit * mono
                for i in range(0, len(pcm), chunk_size):
                    chunk = pcm[i : i + chunk_size]
                    try:
                        await session.send_realtime_input(
                            audio={"data": chunk, "mime_type": "audio/pcm"}
                        )
                    except Exception:
                        break
                    await asyncio.sleep(0.02)  # small pacing delay

            async def receive():
                try:
                    async with asyncio.timeout(45):
                        async for response in session.receive():
                            sc = response.server_content
                            if not sc:
                                continue
                            if sc.input_transcription and sc.input_transcription.text:
                                self.transcript_queue.put(sc.input_transcription.text)
                            if sc.turn_complete:
                                break
                except asyncio.TimeoutError:
                    pass

            send_task = asyncio.create_task(send_audio())
            try:
                await receive()
            finally:
                send_task.cancel()
