"""
conversation_manager.py

Saves each recorded conversation as two plain text files on disk:

    conversations/<PersonName>/20260926_143210.txt           (raw transcript)
    conversations/<PersonName>/20260926_143210_summary.txt   (AI/local summary)

and returns a small metadata dict that the main app stores inside its
existing pickle database, so "last conversation" lookups are instant
without re-reading files from disk.
"""

import os
import datetime

CONVERSATIONS_DIR = "conversations"


def _safe_name(name):
    """Make a name safe to use as a folder name."""
    return "".join(
        c for c in name if c.isalnum() or c in (" ", "_", "-")
    ).strip() or "Unknown"


def get_person_dir(name):
    person_dir = os.path.join(CONVERSATIONS_DIR, _safe_name(name))
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def save_conversation_files(name, transcript_text, summary_text):
    """
    Writes the transcript and summary to text files and returns a record
    dict describing the saved conversation.
    """
    person_dir = get_person_dir(name)

    timestamp = datetime.datetime.now()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")

    transcript_path = os.path.join(person_dir, f"{stamp}.txt")
    summary_path = os.path.join(person_dir, f"{stamp}_summary.txt")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return {
        "timestamp": timestamp.isoformat(),
        "display_date": timestamp.strftime("%d %b %Y, %I:%M %p"),
        "transcript_file": transcript_path,
        "summary_file": summary_path,
        "summary": summary_text,
    }