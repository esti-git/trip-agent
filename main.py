"""Terminal entry point: ask the user for the trip details, check them, and run the agent.

Run from the project root:  python main.py
(For the web interface, see app.py.)
"""
import datetime
import json
import sys

import truststore

# Use the operating system's certificates for HTTPS. Without this, Python
# rejects connections behind filtering proxies such as NetFree.
truststore.inject_into_ssl()

from agent import AgentError, Event, TripAgent  # noqa: E402  (must come after inject_into_ssl)
from config import (MAX_DRIVE_MINUTES, MIN_DRIVE_MINUTES, ConfigError,  # noqa: E402
                    create_llm_client)
from prompts import NO_PREFERENCE_LABEL, TRIP_TYPES, build_goal  # noqa: E402
from tools.geocoding import find_settlement  # noqa: E402
from tools.weather import MAX_DAYS_AHEAD  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Hebrew output in the Windows console
    print("🗺️  סוכן תכנון טיול יומי\n")

    try:
        origin = _ask("מאיזו עיר יוצאים? ", _parse_origin)
        trip_date = _ask("באיזה תאריך? (DD/MM/YYYY): ", _parse_date)
        max_minutes = _ask("זמן נסיעה מקסימלי לכיוון אחד, בדקות: ", _parse_minutes)
        trip_type = _ask(_trip_type_question(), _parse_trip_type)
    except (KeyboardInterrupt, EOFError):
        print("\nבוטל.")
        return 1

    try:
        agent = TripAgent(create_llm_client(), on_event=print_event)
        print("\n🤖 הסוכן מתחיל לעבוד...\n")
        answer = agent.run(build_goal(origin, trip_date.isoformat(), max_minutes, trip_type))
    except (ConfigError, AgentError) as e:
        print(f"\n❌ {e}")
        return 1
    except KeyboardInterrupt:
        print("\nבוטל.")
        return 1

    print("\n" + "=" * 50 + "\n" + answer)
    return 0


def print_event(event: Event) -> None:
    """Print one agent step, so the user can follow its reasoning."""
    if event.kind == "thought":
        print(f"💭 {event.text}")
    elif event.kind == "warning":
        print(f"⚠️  {event.text}")
    elif event.kind == "tool":
        arguments = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
        icon = "❌" if "error" in event.result else "✅"
        print(f"[{event.step}] 🔧 {event.tool}({arguments})")
        print(f"     {icon} {json.dumps(event.result, ensure_ascii=False)}")


def _ask(question: str, parse):
    """Ask until the answer parses. parse() returns the value or raises ValueError."""
    while True:
        answer = input(question).strip()
        if not sys.stdin.isatty():
            print(answer)  # answers piped from a file are not echoed; show them in the log
        try:
            return parse(answer)
        except ValueError as e:
            print(f"   ⚠️  {e}")


def _parse_origin(text: str) -> dict:
    """Find the starting city on the map, so the agent never plans from a wrong place."""
    origin = find_settlement(text)
    if "error" in origin:
        raise ValueError(origin["error"])
    print(f"   📍 {origin['name']}")
    return origin


def _parse_date(text: str) -> datetime.date:
    try:
        trip_date = datetime.datetime.strptime(text, "%d/%m/%Y").date()
    except ValueError:
        raise ValueError("התאריך צריך להיות בפורמט DD/MM/YYYY, למשל 27/10/2026.") from None
    today = datetime.date.today()
    last_day = today + datetime.timedelta(days=MAX_DAYS_AHEAD)
    if not today <= trip_date <= last_day:
        raise ValueError(f"אפשר לתכנן רק מהיום ועד {last_day:%d/%m/%Y} (טווח התחזית).")
    return trip_date


def _parse_minutes(text: str) -> int:
    try:
        minutes = int(text)
    except ValueError:
        raise ValueError("יש להזין מספר שלם של דקות.") from None
    if not MIN_DRIVE_MINUTES <= minutes <= MAX_DRIVE_MINUTES:
        raise ValueError(f"יש להזין בין {MIN_DRIVE_MINUTES} ל-{MAX_DRIVE_MINUTES} דקות.")
    return minutes


def _trip_type_question() -> str:
    options = "\n".join(f"  {i}. {label}" for i, (label, _) in enumerate(TRIP_TYPES.values(), 1))
    return f"איזה סוג טיול? (Enter = {NO_PREFERENCE_LABEL})\n{options}\nבחירה: "


def _parse_trip_type(text: str) -> str | None:
    """A number from the list, or empty for no preference. Returns a TRIP_TYPES key or None."""
    if not text:
        return None
    keys = list(TRIP_TYPES)
    if not text.isdigit() or not 1 <= int(text) <= len(keys):
        raise ValueError(f"יש לבחור מספר בין 1 ל-{len(keys)}, או Enter אם אין העדפה.")
    return keys[int(text) - 1]


if __name__ == "__main__":
    sys.exit(main())
