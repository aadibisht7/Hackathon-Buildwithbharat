"""
last_met.py

Formats a stored "last seen" timestamp into a short, friendly string:

    "11pm, Yesterday"
    "10am, Today"
    "2pm, 12th September 2026"

Accepts either a datetime.datetime or an ISO-format string (the format
already used elsewhere in this app, e.g. conversation_manager.py).
"""

import datetime


def _ordinal(day):
    if 11 <= (day % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _format_time(dt):
    """e.g. '11pm', '2pm', '10:15am' (minutes only shown if non-zero)."""
    hour = dt.hour % 12
    if hour == 0:
        hour = 12
    ampm = "am" if dt.hour < 12 else "pm"

    if dt.minute == 0:
        return f"{hour}{ampm}"

    return f"{hour}:{dt.minute:02d}{ampm}"


def format_last_met(timestamp, now=None):
    """
    Returns a friendly "last met" string, or None if there's nothing to
    show yet (timestamp is None / unparsable).
    """

    if timestamp is None:
        return None

    if isinstance(timestamp, str):
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    else:
        dt = timestamp

    now = now or datetime.datetime.now()

    today = now.date()
    dt_date = dt.date()

    time_part = _format_time(dt)

    if dt_date == today:
        day_part = "Today"
    elif dt_date == today - datetime.timedelta(days=1):
        day_part = "Yesterday"
    else:
        day_part = f"{_ordinal(dt_date.day)} {dt_date.strftime('%B %Y')}"

    return f"{time_part}, {day_part}"


if __name__ == "__main__":
    # Quick manual sanity check: python last_met.py
    now = datetime.datetime(2026, 9, 27, 15, 0, 0)

    samples = [
        now.replace(hour=23, minute=0) - datetime.timedelta(days=1),
        now.replace(hour=10, minute=0),
        datetime.datetime(2026, 9, 12, 14, 0, 0),
    ]

    for s in samples:
        print(s, "->", format_last_met(s, now=now))