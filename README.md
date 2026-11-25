# 🎙️ Real‑Time Speech‑to‑Text (Hindi + English)

A Streamlit web app that transcribes multilingual speech (e.g., Hindi‑English code‑mixing) in real time using the **Gemini Live API**, stores transcripts in a **SQLite** relational database, and lets you manage them via a full **CRUD** dashboard.

---

## ✨ Features

- 🎤 **Real‑time transcription** of mixed Hindi + English speech (auto language detection)
- 🧠 Powered by **Gemini Live API** (`gemini-3.5-transcribe-live`) — open‑source‑friendly, low latency
- 💾 Transcripts persisted in **SQLite** (easily swappable to PostgreSQL/MySQL)
- 📜 **History dashboard** with Create / Read / Update / Delete operations
- ⚡ Non‑blocking background thread — Streamlit UI stays responsive
- 🔐 API key loaded from a **`.env`** file (kept out of version control)

---

## 📁 Project Structure

```
speech-to-text-app/
├── app.py               # Streamlit UI (Live + History tabs)
├── stt_service.py       # Gemini Live API wrapper (background thread)
├── database.py          # SQLite CRUD helpers
├── requirements.txt
├── .env.example         # Template for environment variables
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/speech-to-text-app.git
cd speech-to-text-app
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note (Windows / Linux):** If `pyaudio` fails to install, use:
> - Windows: `pip install pipwin && pipwin install pyaudio`
> - Linux: `sudo apt install portaudio19-dev && pip install pyaudio`

### 3. Configure environment variables

Copy the example file and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a free key at [https://aistudio.google.com/](https://aistudio.google.com/).

> 🔒 **Never commit `.env`** — it is already listed in `.gitignore`.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## 🎤 Usage

1. Ensure `GEMINI_API_KEY` is set in your `.env` file (the sidebar shows whether the key was loaded).
2. Click **▶️ Start Recording** and speak — try mixing languages, e.g.:
   - *"Kal meeting hai at 3 PM."*
   - *"Mujhe ek loan chahiye for my new business."*
3. Transcripts appear **live** below the buttons.
4. Click **⏹️ Stop & Save** — the transcript is written to SQLite.
5. Go to the **📜 History** tab to **edit** or **delete** saved transcripts.

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Gemini API key from Google AI Studio |

Loaded at runtime in `app.py` via `python-dotenv` (`load_dotenv()`), so `.env` is read automatically when the app starts.

---

## 🗄️ Database

Uses **SQLite** out of the box (`transcriptions.db`). Schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto‑increment |
| `text` | TEXT | Transcribed text |
| `language` | TEXT | `auto` by default (or detected) |
| `created_at` | TEXT | ISO‑8601 UTC timestamp |

To swap to PostgreSQL, replace the connection in `database.py` with `psycopg2` or SQLAlchemy.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo.
3. Set main file to `app.py`.
4. In **Settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

   ⚠️ On Streamlit Cloud, `.env` files are **not** used. Either:
   - (a) Use Streamlit Secrets as above and fall back to `st.secrets["GEMINI_API_KEY"]` in code, **or**
   - (b) Set the same value as an **Environment Variable** under *Settings → Secrets → Advanced*.

   A common pattern is:

   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv()

   api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
   ```

5. **Deploy** 🚀

> ⚠️ **Microphone limitation on Streamlit Cloud**  
> The cloud container is **headless** and has no microphone, so `pyaudio` won't work there.  
> For the deployed version, replace the mic capture in `stt_service.py` with a browser‑based recorder (`streamlit-mic-recorder` or `audio-recorder-streamlit`) and forward the audio bytes to `session.send_realtime_input(...)`. Everything else (Gemini Live, queue, CRUD, UI) stays the same.

---

## 🧩 Tech Stack

| Layer | Tech |
|-------|------|
| UI | Streamlit |
| STT | Gemini Live API (`gemini-3.5-transcribe-live`) |
| DB | SQLite |
| Concurrency | `threading` + `asyncio` + `queue.Queue` |
| Config | `python-dotenv` (`.env`) |
| Audio I/O | PyAudio (local) / browser recorder (cloud) |

---

## 🐛 Known Issues / Notes

- `pyaudio` requires native `portaudio` — see install notes above.
- The live session runs in a **background thread** with its own asyncio loop; do **not** call `asyncio.run()` from Streamlit directly (that caused the `_AsyncGeneratorContextManager` error in earlier versions).
- `gemini-3.5-transcribe-live` accepts `language_codes=[]` for **automatic language detection** — essential for Hindi‑English code‑mixing.
- If you see `GEMINI_API_KEY is not set`, check that `.env` exists in the project root and is not committed to git.

---


## 🙏 Acknowledgements

- [Google Gen AI SDK](https://github.com/googleapis/python-genai)
- [Gemini Live API docs](https://ai.google.dev/gemini-api/docs/live)
- [Streamlit](https://streamlit.io)
- [python-dotenv](https://github.com/theskumar/python-dotenv)