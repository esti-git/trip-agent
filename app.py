"""Web interface for the trip agent, built with Streamlit.

Run from the project root:  streamlit run app.py
The agent's steps appear live while it works, then the plan is shown.
"""
import datetime

import streamlit as st
import truststore

# Use the operating system's certificates for HTTPS (see main.py).
truststore.inject_into_ssl()

from agent import AgentError, Event, TripAgent  # noqa: E402  (must come after inject_into_ssl)
from config import ConfigError, create_llm_client  # noqa: E402
from prompts import build_goal  # noqa: E402
from tools.geocoding import geocode  # noqa: E402
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

# Hebrew text right to left, applied only to text elements.
RTL_STYLE = """
<style>
  [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"], [data-testid="stTextInput"] input,
  [data-testid="stMetric"], [data-testid="stExpander"] summary
  { direction: rtl; text-align: right; }
  .stElementContainer:has(> [data-testid="stButtonGroup"]) { align-self: flex-end; }
  /* English tool errors stay readable inside a Hebrew line. */
  [data-testid="stMarkdownContainer"] code { direction: ltr; unicode-bidi: isolate; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="סוכן טיולים", page_icon="🗺️")
    st.markdown(RTL_STYLE, unsafe_allow_html=True)
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
        submitted = st.form_submit_button("🚀 תכנן לי טיול", type="primary",
                                          use_container_width=True)

    if submitted:
        plan_trip(city.strip(), trip_date, max_minutes)


def plan_trip(city: str, trip_date: datetime.date, max_minutes: int | None) -> None:
    """Check the input, run the agent with live steps, and show the result."""
    if not city:
        st.warning("יש להזין עיר מוצא.")
        return
    if max_minutes is None:  # the user clicked the selected option again, clearing it
        st.warning("יש לבחור זמן נסיעה מקסימלי.")
        return
    origin = geocode(city)
    if "error" in origin:
        st.error(f"לא הצלחתי לאתר את '{city}' על המפה. בדקו את השם ונסו שוב.")
        return

    events: list[Event] = []
    with st.status(f"🤖 הסוכן מתכנן טיול מ{origin['name']}...", expanded=True) as status:
        def show_event(event: Event) -> None:
            events.append(event)
            st.markdown(_describe(event))

        try:
            agent = TripAgent(create_llm_client(), on_event=show_event)
            answer = agent.run(build_goal(origin, trip_date.isoformat(), max_minutes))
        except (ConfigError, AgentError) as e:
            status.update(label="הסוכן לא הצליח לעבוד", state="error")
            st.error(str(e))
            return
        checks = sum(1 for e in events if e.kind == "tool")
        status.update(label=f"✅ הסוכן סיים אחרי {checks} בדיקות", state="complete")

    _show_summary_cards(events)
    with st.container(border=True):
        st.markdown(answer.replace("\n", "  \n"))  # keep the answer's line breaks


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


def _show_summary_cards(events: list[Event]) -> None:
    """Big numbers from the last successful weather, drive and schedule checks."""
    def last(tool: str, condition=lambda r: True) -> dict | None:
        matches = [e.result for e in events
                   if e.tool == tool and "error" not in e.result and condition(e.result)]
        return matches[-1] if matches else None

    weather = last("get_weather")
    drive = last("travel_time", lambda r: r["within_limit"])
    schedule = last("plan_schedule", lambda r: r["fits_before_sunset"])
    if not (weather and drive and schedule):
        return  # no complete plan, so the agent's explanation is enough

    cards = reversed(st.columns(4))  # columns run left to right; Hebrew reads from the right
    for card, (label, value) in zip(cards, [
        ("🌡️ טמפרטורה", f"{weather['temp_max_c']}°"),
        ("🌧️ סיכוי לגשם", f"{weather['rain_chance_percent']}%"),
        ("🚗 נסיעה", f"{drive['estimated_minutes']} דק'"),
        ("🌅 שקיעה", weather["sunset"]),
    ]):
        card.metric(label, value)


main()
