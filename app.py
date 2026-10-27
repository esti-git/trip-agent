"""Web interface for the trip agent, built with Streamlit.

Run from the project root:  streamlit run app.py
The agent's steps appear live while it works, then the plan is shown.
"""
import datetime
from html import escape

import streamlit as st
import truststore

# Use the operating system's certificates for HTTPS (see main.py).
truststore.inject_into_ssl()

from agent import AgentError, Event, TripAgent  # noqa: E402  (must come after inject_into_ssl)
from config import ConfigError, create_llm_client  # noqa: E402
from prompts import NO_PREFERENCE_LABEL, TRIP_TYPES, build_goal  # noqa: E402
from tools.geocoding import find_settlement  # noqa: E402
from tools.weather import MAX_DAYS_AHEAD  # noqa: E402

TOOL_LABELS = {
    "geocode": "📍 איתור מקום",
    "get_weather": "🌤️ תחזית",
    "travel_time": "🚗 זמן נסיעה",
    "plan_schedule": "🕗 לוח זמנים",
}

# Choices for the maximum one-way drive. Buttons instead of a slider: Streamlit's
# slider draws its handle in the wrong place when the browser language is Hebrew.
DRIVE_OPTIONS = [30, 45, 60, 90, 120, 180]

NO_PREFERENCE = "none"
TRIP_TYPE_OPTIONS = [NO_PREFERENCE, *TRIP_TYPES]

# The line prefixes of the agent's final answer (see prompts.SYSTEM_PROMPT).
ANSWER_EMOJIS = ("🌤️", "🎯", "📍", "🕗", "💡")

# Hebrew text right to left, applied only to text elements, and the plan card.
STYLE = """
<style>
  .plan-card { direction: rtl; text-align: center; margin: 8px 0 12px;
               padding: 28px 24px; border-radius: 18px;
               background: rgba(15, 118, 110, 0.06); border: 1px solid rgba(15, 118, 110, 0.22); }
  .plan-card .eyebrow { font-size: 0.8rem; letter-spacing: 0.08em; opacity: 0.6; }
  .plan-card .destination { font-size: 2.1rem; font-weight: 700; color: #0f766e;
                            line-height: 1.25; margin: 4px 0 10px; }
  .plan-card .reason { max-width: 520px; margin: 0 auto 18px; opacity: 0.85; line-height: 1.6; }
  .plan-card .facts { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px;
                      margin-bottom: 22px; }
  .plan-card .fact { font-size: 0.85rem; padding: 4px 12px; border-radius: 999px;
                     background: rgba(127, 127, 127, 0.12); }
  .plan-card .timeline { display: flex; justify-content: center; align-items: flex-start;
                         gap: 14px; margin: 0 auto 18px; }
  .plan-card .stop { display: flex; flex-direction: column; gap: 2px; min-width: 64px; }
  .plan-card .stop b { font-size: 1.05rem; }
  .plan-card .stop span { font-size: 0.75rem; opacity: 0.6; }
  .plan-card .arrow { opacity: 0.35; padding-top: 2px; }
  .plan-card .tip { font-size: 0.9rem; opacity: 0.8; margin: 0; }
  @media (prefers-color-scheme: dark) { .plan-card .destination { color: #2dd4bf; } }
</style>
"""

RTL_STYLE = """
<style>
  [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"], [data-testid="stTextInput"] input,
  [data-testid="stMetric"], [data-testid="stExpander"] summary
  { direction: rtl; text-align: right; }
  .stElementContainer:has(> [data-testid="stButtonGroup"]) { align-self: flex-end; }
  [data-testid="stButtonGroup"] { direction: rtl; }  /* first option on the right */
  /* English tool errors stay readable inside a Hebrew line. */
  [data-testid="stMarkdownContainer"] code { direction: ltr; unicode-bidi: isolate; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="סוכן טיולים", page_icon="🗺️")
    st.markdown(RTL_STYLE + STYLE, unsafe_allow_html=True)
    st.title("🗺️ סוכן תכנון טיול יומי")
    st.caption("ספרו לי מאיפה יוצאים ומתי, ואני אבדוק את מזג האוויר, אבחר יעד מתאים "
               "ואבנה לוח זמנים שחוזר לפני השקיעה.")

    today = datetime.date.today()
    with st.form("trip"):
        city = st.text_input("מאיזו עיר יוצאים?", placeholder="למשל: ירושלים")
        trip_date = st.date_input("באיזה תאריך?", value=today, min_value=today,
                                  max_value=today + datetime.timedelta(days=MAX_DAYS_AHEAD),
                                  format="DD/MM/YYYY")
        max_minutes = st.segmented_control("זמן נסיעה מקסימלי לכיוון אחד", DRIVE_OPTIONS,
                                           default=60, format_func=lambda m: f"{m} דק'")
        trip_type = st.segmented_control(
            "איזה סוג טיול? (אם מזג האוויר לא יתאים, הסוכן יציע חלופה ויסביר)",
            TRIP_TYPE_OPTIONS, default=NO_PREFERENCE,
            format_func=lambda key: TRIP_TYPES[key][0] if key in TRIP_TYPES else NO_PREFERENCE_LABEL)
        submitted = st.form_submit_button("🚀 תכנן לי טיול", type="primary",
                                          use_container_width=True)

    if submitted:
        # A cleared choice (clicking the selected option again) also means no preference.
        preference = trip_type if trip_type in TRIP_TYPES else None
        plan_trip(city.strip(), trip_date, max_minutes, preference)


def plan_trip(city: str, trip_date: datetime.date, max_minutes: int | None,
              trip_type: str | None) -> None:
    """Check the input, run the agent with live steps, and show the result."""
    if not city:
        st.warning("יש להזין עיר מוצא.")
        return
    if max_minutes is None:  # the user clicked the selected option again, clearing it
        st.warning("יש לבחור זמן נסיעה מקסימלי.")
        return
    origin = find_settlement(city)
    if "error" in origin:
        st.error(origin["error"])
        return

    events: list[Event] = []
    with st.status(f"🤖 הסוכן מתכנן טיול מ{origin['name']}...", expanded=True) as status:
        def show_event(event: Event) -> None:
            events.append(event)
            st.markdown(_describe(event))

        try:
            agent = TripAgent(create_llm_client(), on_event=show_event)
            answer = agent.run(build_goal(origin, trip_date.isoformat(), max_minutes, trip_type),
                               verified_places=[origin])
        except (ConfigError, AgentError) as e:
            status.update(label="הסוכן לא הצליח לעבוד", state="error")
            st.error(str(e))
            return
        checks = sum(1 for e in events if e.kind == "tool")
        status.update(label=f"✅ הסוכן סיים אחרי {checks} בדיקות", state="complete")

    show_result(answer, events)


def _describe(event: Event) -> str:
    """One readable line for an agent step."""
    if event.kind == "thought":
        return f"💭 {event.text}"
    if event.kind == "warning":
        return f"⚠️ {event.text}"

    label = TOOL_LABELS.get(event.tool, event.tool)
    r = event.result
    if "error" in r:
        return f"**{label}** ❌ `{r['error']}`"
    if event.tool == "geocode":
        summary = r["name"]
    elif event.tool == "get_weather":
        summary = f"{r['temp_max_c']}° / {r['temp_min_c']}°, סיכוי לגשם {r['rain_chance_percent']}%"
    elif event.tool == "travel_time":
        verdict = "בטווח" if r["within_limit"] else "מחוץ לטווח"
        summary = f"{r['estimated_minutes']} דק' ({r['estimated_road_km']} ק\"מ), {verdict}"
    elif event.tool == "plan_schedule":
        verdict = "לפני השקיעה" if r["fits_before_sunset"] else f"חורג ב-{-r['spare_minutes']} דק'"
        summary = f"{r['leave_home']} ← {r['back_home']}, {verdict}"
    else:
        summary = str(r)
    return f"**{label}** ✅ {summary}"


def show_result(answer: str, events: list[Event]) -> None:
    """Show the plan as a card centered on the destination, or the agent's explanation."""
    lines = _answer_lines(answer)
    weather = _last_result(events, "get_weather")
    drive = _last_result(events, "travel_time", lambda r: r["within_limit"])
    schedule = _last_result(events, "plan_schedule", lambda r: r["fits_before_sunset"])

    if not ("📍" in lines and weather and drive and schedule):
        # No complete plan: the agent explains what it tried and why it did not work.
        with st.container(border=True):
            st.markdown(answer.replace("\n", "  \n"))
        return

    st.markdown(_plan_card_html(lines, weather, drive, schedule), unsafe_allow_html=True)
    with st.expander("התשובה המלאה של הסוכן"):
        st.markdown(answer.replace("\n", "  \n"))  # keep the answer's line breaks


def _answer_lines(answer: str) -> dict[str, str]:
    """Split the agent's five-line answer into {emoji: text}, e.g. {"📍": "חוף גינוסר (9 ק"מ...)"}."""
    lines = {}
    for line in answer.splitlines():
        for emoji in ANSWER_EMOJIS:
            if line.strip().startswith(emoji) and ":" in line:
                lines[emoji] = line.split(":", 1)[1].strip()
    return lines


def _last_result(events: list[Event], tool: str, condition=lambda r: True) -> dict | None:
    """The last successful result of a tool that meets the condition."""
    matches = [e.result for e in events
               if e.tool == tool and "error" not in e.result and condition(e.result)]
    return matches[-1] if matches else None


def _plan_card_html(lines: dict, weather: dict, drive: dict, schedule: dict) -> str:
    """The plan card: destination in the center, small facts below it, then the day's timeline."""
    destination = lines["📍"].split("(")[0].strip()
    facts = [f"🌡️ {weather['temp_max_c']}°", f"🌧️ {weather['rain_chance_percent']}%",
             f"🚗 {drive['estimated_minutes']} דק'", f"🌅 {weather['sunset']}"]
    timeline = [("יציאה", schedule["leave_home"]), ("הגעה", schedule["arrive"]),
                ("יציאה חזרה", schedule["leave_place"]), ("בבית", schedule["back_home"])]

    facts_html = "".join(f'<span class="fact">{escape(f)}</span>' for f in facts)
    timeline_html = '<span class="arrow">←</span>'.join(
        f'<div class="stop"><b>{escape(time)}</b><span>{label}</span></div>' for label, time in timeline)
    reason = f'<p class="reason">{escape(lines["🎯"])}</p>' if "🎯" in lines else ""
    tip = f'<p class="tip">💡 {escape(lines["💡"])}</p>' if "💡" in lines else ""
    return (f'<div class="plan-card"><div class="eyebrow">היעד שלך</div>'
            f'<div class="destination">{escape(destination)}</div>{reason}'
            f'<div class="facts">{facts_html}</div>'
            f'<div class="timeline">{timeline_html}</div>{tip}</div>')


if __name__ == "__main__":  # Streamlit runs this file as __main__; importing it does not
    main()
