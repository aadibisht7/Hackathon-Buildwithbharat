"""
speech_recorder.py

Background microphone listener that converts speech to text in real time.

Uses the `speech_recognition` package with Google's free web speech API
(no API key needed, but requires an internet connection). If you need
fully offline recognition, swap `recognize_google` for a Vosk-based
recognizer inside `_callback`.

Install:
    pip install SpeechRecognition pyaudio
"""

import speech_recognition as sr


class SpeechRecorder:
    """
    Records from the default microphone in the background and turns
    speech into text incrementally, until `stop()` is called.
    """

    def __init__(self, on_partial_text=None):
        """
        on_partial_text: optional callback(str) called every time a new
                         chunk of speech is transcribed. Useful for showing
                         a live transcript while recording.
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.on_partial_text = on_partial_text

        self.full_transcript = []
        self._stop_listening = None
        self.is_recording = False

    def start(self):
        """Begin listening in the background. Raises on microphone errors."""
        self.full_transcript = []
        self.is_recording = True

        with self.microphone as source:
            # Quick calibration so background noise isn't picked up as speech
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        self._stop_listening = self.recognizer.listen_in_background(
            self.microphone,
            self._callback,
            phrase_time_limit=8
        )

    def _callback(self, recognizer, audio):
        if not self.is_recording:
            return

        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            # Nothing intelligible in this chunk - just skip it
            return
        except sr.RequestError as e:
            print("Speech recognition service error:", e)
            return

        if text:
            self.full_transcript.append(text)
            if self.on_partial_text:
                self.on_partial_text(text)

    def stop(self):
        """
        Stop listening and return the full transcript collected so far
        as a single string.
        """
        self.is_recording = False

        if self._stop_listening is not None:
            self._stop_listening(wait_for_stop=False)
            self._stop_listening = None

        return " ".join(self.full_transcript).strip()