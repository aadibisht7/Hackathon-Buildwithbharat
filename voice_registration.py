"""
voice_registration.py

Lets a family member register themselves by speaking a short introduction,
e.g. "Hello, I am Rahul, your son" or "My name is Priya, I'm your daughter".

`register_from_voice()` listens on the microphone, transcribes the speech,
and tries to pull out a name and a relationship from it. Returns:

    {"name": "Rahul", "relationship": "son"}

or `None` if nothing usable was heard/understood.

Install:
    pip install SpeechRecognition pyaudio
"""

import re
import speech_recognition as sr


# ============================================================
# Known relationship words we try to recognize
# ============================================================

KNOWN_RELATIONSHIPS = {
    "son", "daughter", "wife", "husband", "mother", "father",
    "mom", "dad", "mommy", "daddy",
    "sister", "brother", "grandson", "granddaughter",
    "grandma", "grandpa", "grandmother", "grandfather",
    "uncle", "aunt", "cousin", "nephew", "niece",
    "friend", "neighbor", "neighbour", "nurse", "caregiver", "doctor",
}

# Words to ignore when guessing a name from the fallback parser
_STOPWORDS = {
    "hello", "hi", "hey", "hiya",
    "i", "im", "i'm", "am", "is", "my", "name", "names", "this",
    "your", "and", "the", "a", "an", "to", "from", "call", "me",
    "here", "it's", "its", "that's", "thats",
}


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def register_from_voice(timeout=8, phrase_time_limit=12):
    """
    Listens for one spoken phrase and returns {"name", "relationship"},
    or None if nothing could be captured/understood/parsed.
    """

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )

    except sr.WaitTimeoutError:
        print("Voice registration: no speech detected in time.")
        return None

    except Exception as e:
        print("Voice registration: microphone error:", e)
        return None

    try:
        text = recognizer.recognize_google(audio)

    except sr.UnknownValueError:
        print("Voice registration: could not understand audio.")
        return None

    except sr.RequestError as e:
        print("Voice registration: speech service error:", e)
        return None

    print(f'Voice registration heard: "{text}"')

    parsed = parse_registration_text(text)

    if parsed is None:
        print("Voice registration: could not find a name/relationship in:", text)
        return None

    return parsed


# ============================================================
# PARSING
# ============================================================

def parse_registration_text(text):
    """
    Tries a series of common phrasings first, then falls back to a
    best-effort word-scan. Returns {"name", "relationship"} or None.
    """

    lowered = text.lower().strip()

    # ------------------------------------------------------------
    # Pattern 1: "I am Rahul, your son" / "I'm Rahul your son"
    # ------------------------------------------------------------
    match = re.search(
        r"\bi(?:'?m| am)\s+([a-z]+)\s*,?\s*your\s+([a-z]+)\b",
        lowered
    )
    if match:
        return _build_result(match.group(1), match.group(2))

    # ------------------------------------------------------------
    # Pattern 2: "This is Rahul, your son"
    # ------------------------------------------------------------
    match = re.search(
        r"\bthis is\s+([a-z]+)\s*,?\s*your\s+([a-z]+)\b",
        lowered
    )
    if match:
        return _build_result(match.group(1), match.group(2))

    # ------------------------------------------------------------
    # Pattern 3: "My name is Rahul, I am your son"
    #            "My name is Priya and I'm your daughter"
    # ------------------------------------------------------------
    match = re.search(
        r"\bmy name is\s+([a-z]+)\b[^a-z]*(?:and\s+)?i(?:'?m| am)\s+your\s+([a-z]+)\b",
        lowered
    )
    if match:
        return _build_result(match.group(1), match.group(2))

    # ------------------------------------------------------------
    # Pattern 4: "Call me Rahul, your son"
    # ------------------------------------------------------------
    match = re.search(
        r"\bcall me\s+([a-z]+)\s*,?\s*your\s+([a-z]+)\b",
        lowered
    )
    if match:
        return _build_result(match.group(1), match.group(2))

    # ------------------------------------------------------------
    # Pattern 5: "Rahul is your son" (name first, no "I am")
    # ------------------------------------------------------------
    match = re.search(
        r"\b([a-z]+)\s+is your\s+([a-z]+)\b",
        lowered
    )
    if match:
        return _build_result(match.group(1), match.group(2))

    # ------------------------------------------------------------
    # Pattern 6: "Your son, Rahul" / "your daughter Priya"
    #            (relationship spoken before the name)
    # ------------------------------------------------------------
    match = re.search(
        r"\byour\s+([a-z]+)\s*,?\s+([a-z]+)\b",
        lowered
    )
    if match and match.group(1) in KNOWN_RELATIONSHIPS:
        return _build_result(match.group(2), match.group(1))

    # ------------------------------------------------------------
    # Fallback: scan for any known relationship word, then guess the
    # name as the first remaining "real" word in the sentence.
    # ------------------------------------------------------------
    tokens = re.findall(r"[a-zA-Z']+", text)

    relationship = next(
        (t.lower() for t in tokens if t.lower() in KNOWN_RELATIONSHIPS),
        None
    )

    if relationship is None:
        return None

    name = next(
        (
            t for t in tokens
            if t.lower() not in _STOPWORDS
            and t.lower() not in KNOWN_RELATIONSHIPS
        ),
        None
    )

    if name is None:
        return None

    return _build_result(name, relationship)


def _build_result(name, relationship):
    name = name.strip().title()
    relationship = relationship.strip().lower()

    if not name or not relationship:
        return None

    return {"name": name, "relationship": relationship}


# ============================================================
# QUICK MANUAL TEST
# ============================================================

if __name__ == "__main__":
    # Lets you sanity-check the parser without a microphone:
    #   python voice_registration.py
    samples = [
        "Hello, I am Rahul, your son",
        "Hi, I'm Priya your daughter",
        "This is Amit, your brother",
        "My name is Sunita and I am your wife",
        "Call me Vikram, your grandson",
        "Arjun is your nephew",
        "Your son, Karan",
    ]

    for sample in samples:
        print(f"{sample!r:45} -> {parse_registration_text(sample)}")