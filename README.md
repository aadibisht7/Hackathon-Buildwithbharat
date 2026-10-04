# ForgetMeNot — Dementia Assistant

ForgetMeNot is a desktop application designed to assist people with dementia by recognizing familiar people, displaying when they were last seen, and preserving conversations for later recall.

The application combines real-time face recognition, voice registration, speech-to-text, and AI-powered conversation summarization in a single desktop interface.

## Features

* Real-time face recognition using DeepFace and ArcFace
* Voice-based registration of new people
* Name and relationship extraction from spoken introductions
* Conversation recording and speech-to-text transcription
* AI-generated conversation summaries using Google Gemini
* Local fallback summaries when Gemini is unavailable
* "Last Met" information with human-readable timestamps
* Local storage of registered people and conversation history
* Tkinter-based desktop interface

## How It Works

### Face Recognition

The webcam continuously captures frames and detects faces. Registered faces are converted into embeddings and compared against the stored face data.

When a match is found, ForgetMeNot displays information about the recognized person, including their name, relationship, and last-seen time.

### Voice Registration

A new person can register themselves by speaking a short introduction:

```text
Hi, I'm Rahul, your son.
```

The application extracts the person's name and relationship and associates them with their registered face.

### Conversation Memory

When a conversation is recorded, ForgetMeNot:

1. Records audio through the microphone.
2. Converts speech into text.
3. Associates the conversation with the recognized person.
4. Generates a short summary.
5. Stores the transcript and summary locally.

## Architecture

```text
                    Webcam
                       |
                       v
                Face Detection
                       |
                       v
                Face Embedding
                  DeepFace / ArcFace
                       |
                       v
                 Face Matching
                       |
                       v
              Recognized Person
                       |
              +--------+--------+
              |                 |
              v                 v
          Last Met        Conversation
                              |
                              v
                       Speech-to-Text
                              |
                              v
                         Transcript
                              |
                              v
                       Gemini Summary
                              |
                              v
                       Saved Memory
```

## Tech Stack

| Technology        | Purpose                             |
| ----------------- | ----------------------------------- |
| Python            | Core application                    |
| Tkinter           | Desktop interface                   |
| OpenCV            | Webcam capture and image processing |
| DeepFace          | Face recognition                    |
| ArcFace           | Face embedding model                |
| NumPy             | Numerical operations                |
| Pillow            | Image processing                    |
| SpeechRecognition | Speech-to-text                      |
| PyAudio           | Microphone input                    |
| Google Gemini     | Conversation summarization          |
| python-dotenv     | Environment variable management     |
| Pickle            | Local storage of face data          |

## Project Structure

```text
ForgetMeNot/
│
├── dementia_assistant_pro.py
├── conversation_manager.py
├── last_met.py
├── speech_recorder.py
├── summarizer.py
├── voice_registration.py
│
├── conversations/
│   └── <PersonName>/
│       ├── transcript files
│       └── summary files
│
├── family_deepface_data.pkl
├── .env
└── .gitignore
```

### Main Files

| File                        | Description                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `dementia_assistant_pro.py` | Main application, Tkinter UI, camera handling, face recognition, and application workflow |
| `conversation_manager.py`   | Stores conversation transcripts and summaries                                             |
| `last_met.py`               | Formats timestamps into readable "last met" information                                   |
| `speech_recorder.py`        | Handles microphone recording and speech-to-text                                           |
| `summarizer.py`             | Generates Gemini summaries and handles the local fallback                                 |
| `voice_registration.py`     | Processes spoken introductions and extracts name and relationship                         |

## Requirements

* Python 3.9+
* Working webcam
* Working microphone
* Internet connection for speech recognition
* Internet connection for Gemini summarization
* Windows is currently recommended

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ForgetMeNot
```

Install the required dependencies:

```bash
pip install opencv-python numpy pillow deepface SpeechRecognition pyaudio python-dotenv google-genai
```

### PyAudio on Windows

If PyAudio installation fails, try:

```bash
pip install pipwin
pipwin install pyaudio
```

## Gemini Configuration

Gemini is used to generate concise summaries of recorded conversations.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_SUMMARY_MODEL=gemini-3.5-flash
```

The application loads the `.env` file automatically using `python-dotenv`.

If Gemini is unavailable because of an invalid key, network problem, quota limit, or API error, the application falls back to a simple local summary.

Do not commit your API key to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
```

## Running the Application

Run:

```bash
python dementia_assistant_pro.py
```

On startup, the application:

1. Opens the configured webcam.
2. Initializes the face-recognition system.
3. Loads previously registered people.
4. Starts looking for faces.
5. Allows new people to be registered.
6. Allows conversations to be recorded and processed.

## Camera Configuration

The current Windows configuration uses camera index `1` and falls back to `0`.

```python
cv2.VideoCapture(1, cv2.CAP_DSHOW)
```

If your webcam uses a different index, update the camera configuration in `dementia_assistant_pro.py`.

For macOS or Linux, the camera initialization may need to be modified.

## Data Storage

ForgetMeNot currently uses local storage.

Registered face data is stored in:

```text
family_deepface_data.pkl
```

Conversation data is stored under:

```text
conversations/
```

Example:

```text
conversations/
├── Rahul/
│   ├── conversation_01.txt
│   ├── summary_01.txt
│   └── ...
│
└── Priya/
    ├── conversation_01.txt
    └── ...
```

## Privacy

The application processes personal information including face embeddings, names, relationships, voice input, transcripts, and conversation summaries.

For development purposes, this information is stored locally.

The following files should not be committed to a public repository:

```text
.env
family_deepface_data.pkl
conversations/
```

Recommended `.gitignore`:

```gitignore
.env
family_deepface_data.pkl
conversations/
__pycache__/
*.pyc
```

## Current Limitations

* Speech recognition requires an internet connection.
* Gemini summarization requires a valid API key and network access.
* Face recognition accuracy depends on lighting, camera quality, face angle, and recognition thresholds.
* The current camera configuration is primarily designed for Windows.
* Local conversation storage is not currently encrypted.
* The project is currently a prototype and has not been designed for clinical use.

## Future Improvements

* Improved face tracking and recognition stability
* Better multi-person recognition
* Searchable conversation memories
* Voice playback of saved memories
* Offline speech-to-text
* Offline summarization
* Encrypted local storage
* Secure caregiver/family dashboard
* Memory timeline
* Automatic conversation categorization
* Mobile companion application
* Improved accessibility features

## Project Goal

ForgetMeNot aims to provide a simple memory-assistance tool that helps users recognize familiar people and recall meaningful interactions.

The goal is not to replace human interaction, but to use technology as a support layer around it.

## Contributing

Contributions and suggestions are welcome.

For larger changes, open an issue before submitting a pull request.

## License

This project is currently intended for educational, experimental, and hackathon development.

Add an app
