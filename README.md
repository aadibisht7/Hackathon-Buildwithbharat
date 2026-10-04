# ForgetMeNot — Dementia Assistant

A desktop app that uses a webcam to recognize registered family members / caregivers in real time, shows "last met" info, and can record, transcribe, and summarize conversations for later recall.

## Features

- **Face recognition** (via DeepFace / ArcFace) to identify registered people from the webcam feed.
- **Voice registration** — a new person can register themselves by speaking a short intro (e.g. *"Hi, I'm Rahul, your son"*).
- **Conversation recording** — records speech in the background, transcribes it, and stores a raw transcript + AI-generated summary per person.
- **AI summarization** via Google Gemini (falls back to a simple local summary if no API key is set).
- **"Last Met"** friendly timestamp formatting (e.g. *"11pm, Yesterday"*).

## Project files

| File | Purpose |
|---|---|
| `dementia_assistant_pro.py` | Main application (Tkinter UI, camera loop, face recognition) — run this file |
| `conversation_manager.py` | Saves transcripts/summaries to disk under `conversations/<Name>/` |
| `last_met.py` | Formats "last seen" timestamps |
| `speech_recorder.py` | Background microphone → text transcription |
| `summarizer.py` | Turns a transcript into a short AI (Gemini) or local summary |
| `voice_registration.py` | Parses a spoken introduction into a name + relationship |

## Requirements

- Python 3.9+
- A working webcam and microphone
- Windows is assumed for camera capture (`cv2.CAP_DSHOW`); adjust `cv2.VideoCapture` calls in `dementia_assistant_pro.py` if you're on macOS/Linux.

### Install dependencies

```bash
pip install opencv-python numpy pillow deepface SpeechRecognition pyaudio python-dotenv google-genai
```

> `pyaudio` can be tricky to install on some systems. On Windows, if `pip install pyaudio` fails, try:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

## Setting up the Gemini API key

Conversation summaries use Google's Gemini API. Without a key, the app still works but falls back to a plain "first few sentences" summary instead of an AI-written one.

You have two options — pick whichever fits your workflow. **The `.env` file is the recommended option since it's automatic and persists across sessions.**

### Option 1 (recommended): `.env` file

Create a file named `.env` in the same folder as `dementia_assistant_pro.py` with:

```
GEMINI_API_KEY=your_gemini_key_here
GEMINI_SUMMARY_MODEL=gemini-3.5-flash
```

`summarizer.py` calls `load_dotenv()` on import, so this is picked up automatically every time you run the app — no need to set anything manually in the terminal.

> Note: `GEMINI_SUMMARY_MODEL` isn't currently read by `summarizer.py` (the model name `gemini-3.5-flash` is hardcoded). Setting it in `.env` does no harm, but if you want it to actually control the model, update `_summarize_with_gemini()` to read `os.environ.get("GEMINI_SUMMARY_MODEL", "gemini-3.5-flash")`.

### Option 2: Set environment variables before running (PowerShell)

If you'd rather not use a `.env` file, set the variables in your PowerShell session before launching the app:

```powershell
$env:GEMINI_API_KEY = "Your Gemini Key"
$env:GEMINI_SUMMARY_MODEL = "gemini-3.5-flash"
python dementia_assistant_pro.py
```

These environment variables only last for the current PowerShell session — you'll need to re-run those two lines each time you open a new terminal (unless you use the `.env` file instead, or set them permanently via `setx`).

## Running the app

```bash
python dementia_assistant_pro.py
```

On first launch, the app will:
1. Try to open your webcam (`camera index 1`, falling back to `0`).
2. Warm up the face-recognition model in the background.
3. Show the main window, where you can register people (by voice or manually) and start/stop conversation recordings.

Registered people, their face embeddings, and conversation history are stored in `family_deepface_data.pkl` in the working directory. Raw transcripts and summaries are additionally saved as text files under `conversations/<PersonName>/`.

## Notes / things to double check

- `speech_recorder.py` and `voice_registration.py` use Google's free web speech API (via `SpeechRecognition`), which requires an internet connection even though no separate API key is needed for it.
- If Gemini summarization fails for any reason (bad key, network issue, quota), the app automatically falls back to the local summary and prints the error to the console rather than stopping.
- Keep your `.env` file out of version control (add it to `.gitignore`) since it contains your API key.
