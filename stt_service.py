"""
Speech-to-Text service using the Gemini Live API.
Fully instrumented with logging at every step.
"""

import asyncio
import io
import threading
import queue
import wave
import audioop
import logging

from google import genai
from google.genai import types

log = logging.getLogger("stt_service")

TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLE_WIDTH = 2  # 16-bit
TARGET_CHANNELS = 1

DEFAULT_MODEL = "gemini-3.5-transcribe-live"

# Exit the receive loop after this many seconds of silence
INACTIVITY_SECS = 12
# Hard safety ceiling for a single session
MAX_TOTAL_SECS = 90


def wav_to_pcm_16k_mono(wav_bytes: bytes) -> bytes:
    """Convert arbitrary WAV bytes to 16 kHz / mono / 16-bit PCM."""
    log.info("wav_to_pcm: received %d bytes", len(wav_bytes))
    if len(wav_bytes) == 0:
        raise ValueError("wav_bytes is empty")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        src_rate = wf.getframerate()
        src_channels = wf.getnchannels()
        src_width = wf.getsampwidth()
        src_frames = wf.getnframes()
        frames = wf.readframes(src_frames)

    log.info(
        "wav_to_pcm: source = rate=%d, channels=%d, width=%d, frames=%d, duration=%.2fs",
        src_rate,
        src_channels,
        src_width,
        src_frames,
        src_frames / src_rate,
    )

    if src_channels > 1:
        frames = audioop.tomono(frames, src_width, 0.5, 0.5)
        log.debug("wav_to_pcm: downmixed to mono")

    if src_width != TARGET_SAMPLE_WIDTH:
        frames = audioop.lin2lin(frames, src_width, TARGET_SAMPLE_WIDTH)
        log.debug("wav_to_pcm: converted sample width -> %d", TARGET_SAMPLE_WIDTH)

    if src_rate != TARGET_SAMPLE_RATE:
        frames, _ = audioop.ratecv(
            frames, TARGET_SAMPLE_WIDTH, 1, src_rate, TARGET_SAMPLE_RATE, None
        )
        log.debug("wav_to_pcm: resampled -> %d Hz", TARGET_SAMPLE_RATE)

    log.info(
        "wav_to_pcm: output = %d PCM bytes (%.2fs @ 16kHz)",
        len(frames),
        len(frames) / (TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH),
    )
    return frames


class LiveTranscriber:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, on_complete=None):
        log.info(
            "LiveTranscriber.__init__: model=%s on_complete=%s",
            model, callable(on_complete),
        )
        self.client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )
        self.model = model
        self.config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[],
                mode="VERBATIM",                     # ← add this
            ),
            realtime_input_config=types.RealtimeInputConfig(   # ← add manual VAD
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True,
                ),
            ),
        )
        self.on_complete = on_complete
        self._collected_chunks: list[str] = []
        self.transcript_queue: "queue.Queue[str]" = queue.Queue()
        self.done = threading.Event()
        self.error: str | None = None
        self._thread: "threading.Thread | None" = None

    # ------------------------------------------------------------------ public
    def start(self, wav_bytes: bytes):
        log.info("start: called with %d audio bytes", len(wav_bytes))
        if self._thread and self._thread.is_alive():
            log.warning("start: thread already alive, ignoring call")
            return
        self.done.clear()
        self.error = None
        self._collected_chunks.clear()
        self._thread = threading.Thread(
            target=self._run, args=(wav_bytes,), daemon=True
        )
        self._thread.start()
        log.info("start: background thread launched (id=%s)", self._thread.ident)

    def is_running(self) -> bool:
        running = self._thread is not None and self._thread.is_alive()
        log.debug("is_running: %s", running)
        return running

    def get_chunk(self) -> str | None:
        try:
            chunk = self.transcript_queue.get_nowait()
            log.debug("get_chunk: returned %r", chunk)
            return chunk
        except queue.Empty:
            return None

    # ----------------------------------------------------------------- internal
    def _run(self, wav_bytes: bytes):
        log.info("_run: thread started")
        try:
            asyncio.run(self._transcribe(wav_bytes))
            log.info("_run: _transcribe completed normally")
        except Exception as e:  # noqa: BLE001
            log.exception("_run: unhandled exception")
            self.error = str(e)
        finally:
            self.done.set()
            log.info("_run: done flag set (error=%r)", self.error)

    async def _transcribe(self, wav_bytes: bytes):
        log.info("_transcribe: converting WAV -> PCM")
        pcm = wav_to_pcm_16k_mono(wav_bytes)
        log.info("_transcribe: PCM ready, %d bytes", len(pcm))
        self._collected_chunks.clear()

        log.info("_transcribe: opening Live session (model=%s)", self.model)
        try:
            async with self.client.aio.live.connect(
                model=self.model, config=self.config
            ) as session:
                log.info("_transcribe: ✅ Live session OPENED")

                async def send_audio():
                    chunk_size = 3200
                    n_chunks = (len(pcm) + chunk_size - 1) // chunk_size
                    log.info("send_audio: sending %d chunks (manual VAD)", n_chunks)

                    # 👇 tell the server "speech starts now"
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                    log.info("send_audio: activity_start sent")

                    for i in range(0, len(pcm), chunk_size):
                        idx = i // chunk_size
                        chunk = pcm[i : i + chunk_size]
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk,
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                            if idx % 10 == 0 or idx == n_chunks - 1:
                                log.debug("send_audio: sent chunk %d/%d", idx + 1, n_chunks)
                        except Exception:
                            log.exception("send_audio: send failed at chunk %d", idx)
                            break
                        await asyncio.sleep(0.02)

                    # 👇 tell the server "speech ended — flush and transcribe"
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    log.info("send_audio: activity_end sent")

                    # Optional belt-and-braces: also close the audio stream
                    await session.send_realtime_input(audio_stream_end=True)
                    log.info("send_audio: audio_stream_end=True sent")

                async def receive():
                    log.info("receive: starting to listen for responses")
                    msg_count = 0
                    try:
                        async with asyncio.timeout(MAX_TOTAL_SECS) as t:
                            loop_time = asyncio.get_running_loop().time
                            t.reschedule(loop_time() + INACTIVITY_SECS)

                            async for response in session.receive():
                                # reset inactivity window on every message
                                t.reschedule(loop_time() + INACTIVITY_SECS)
                                msg_count += 1

                                log.info(
                                    "receive: msg #%d | type=%s | "
                                    "has_server_content=%s",
                                    msg_count,
                                    type(response).__name__,
                                    bool(response.server_content),
                                )

                                sc = response.server_content
                                if not sc:
                                    log.debug(
                                        "receive: msg #%d has no "
                                        "server_content, full=%r",
                                        msg_count, response,
                                    )
                                    continue

                                it = getattr(sc, "input_transcription", None)
                                if it is not None:
                                    log.info(
                                        "receive: msg #%d "
                                        "input_transcription=text=%r, "
                                        "finished=%r",
                                        msg_count,
                                        getattr(it, "text", None),
                                        getattr(it, "finished", None),
                                    )
                                    if it.text:
                                        # 👇 key fix: BOTH queue and collected list
                                        self.transcript_queue.put(it.text)
                                        self._collected_chunks.append(it.text)
                                        log.info(
                                            "receive: ✅ queued transcript: %r",
                                            it.text,
                                        )
                                    if getattr(it, "finished", False):
                                        log.info(
                                            "receive: "
                                            "input_transcription.finished=True "
                                            "— stopping"
                                        )
                                        break
                                else:
                                    log.debug(
                                        "receive: msg #%d no "
                                        "input_transcription",
                                        msg_count,
                                    )

                                if getattr(sc, "turn_complete", False):
                                    log.info(
                                        "receive: turn_complete received — stopping"
                                    )
                                    break

                    except asyncio.TimeoutError:
                        log.info(
                            "receive: exited via inactivity timeout "
                            "(%.0fs of silence) after %d messages",
                            INACTIVITY_SECS, msg_count,
                        )
                    except Exception:
                        log.exception(
                            "receive: loop error (msg_count=%d)", msg_count
                        )
                    finally:
                        log.info(
                            "receive: exited, total messages=%d, queued=%d, "
                            "collected=%d",
                            msg_count,
                            self.transcript_queue.qsize(),
                            len(self._collected_chunks),
                        )

                send_task = asyncio.create_task(send_audio())
                try:
                    await receive()
                finally:
                    if not send_task.done():
                        send_task.cancel()
                        try:
                            await send_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    log.info("_transcribe: session closing")

            log.info("_transcribe: ✅ session closed cleanly")
        finally:
            # 👇 Persist the transcript regardless of UI / session state
            if self.on_complete and self._collected_chunks:
                full_text = " ".join(self._collected_chunks).strip()
                if full_text:
                    try:
                        log.info(
                            "_transcribe: calling on_complete with %d chars",
                            len(full_text),
                        )
                        self.on_complete(full_text)
                        log.info("_transcribe: ✅ on_complete succeeded")
                    except Exception:
                        log.exception("_transcribe: on_complete failed")
            else:
                log.warning(
                    "_transcribe: no chunks collected (collected=%d, "
                    "on_complete=%s)",
                    len(self._collected_chunks),
                    callable(self.on_complete),
                )