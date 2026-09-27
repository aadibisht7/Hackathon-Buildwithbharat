"""
summarizer.py

Turns a raw conversation transcript into a short, warm summary.

Uses Gemini when GEMINI_API_KEY or GOOGLE_API_KEY is available.
Otherwise, falls back to a simple local summary.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()


def summarize_text(text, name="the person", max_sentences=4):
    """
    Summarize a conversation transcript.

    Parameters:
        text: Raw conversation transcript
        name: Name of the person the conversation was with
        max_sentences: Number of sentences for local fallback
    """

    if not text or not text.strip():
        return "No conversation was captured."

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    if api_key:
        try:
            return _summarize_with_gemini(
                text,
                name,
                api_key
            )

        except Exception as e:
            print(
                "AI summarization failed, "
                "falling back to local summary:",
                repr(e)
            )

    return _simple_summary(text, max_sentences)


def _summarize_with_gemini(text, name, api_key):
    """
    Generate an AI summary using Gemini.
    """

    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = (
        "You are helping a caregiver app for a person with dementia. "

        "Summarize the following conversation transcript in "
        "3-5 short, warm, plain-language sentences. "

        f"Write the summary in the THIRD PERSON. "
        f"Refer to the person as \"{name}\". "

        f"For example: \"{name} talked about...\" or "
        f"\"{name} mentioned...\". "

        "Never address the reader as 'you'. "

        "Mention who else was involved and what topics came up. "

        "Mention anything important that may need to be remembered "
        "later. "

        "Do not invent details that are not present in the transcript. "

        "Do not include a preamble. "
        "Return only the summary.\n\n"

        f"Transcript:\n{text}"
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


def _simple_summary(text, max_sentences=4):
    """
    Lightweight offline fallback.
    Keeps the first few sentences.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    sentences = [
        sentence
        for sentence in sentences
        if sentence
    ]

    if not sentences:
        return text.strip()

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    return " ".join(sentences[:max_sentences]) + " ..."