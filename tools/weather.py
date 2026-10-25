"""get_weather tool: the daily forecast for one place and date, using Open-Meteo.

The agent reads this forecast to decide what kind of trip fits the day
(indoor, water and shade, or a view), and uses the sunset time to plan
a safe return.
"""
import datetime

from tools.geocoding import check_in_israel
from tools.http_utils import get_json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_DAYS_AHEAD = 15  # Open-Meteo forecasts today plus the next 15 days
DAILY_FIELDS = ("temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                "precipitation_sum,wind_speed_10m_max,uv_index_max,sunrise,sunset")

GET_WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather forecast for a location on a given date: "
                       "temperatures, chance of rain, wind, UV index, sunrise and sunset. "
                       "Works only for today up to 15 days ahead.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
            },
            "required": ["lat", "lon", "date"],
        },
    },
}


def get_weather(lat: float, lon: float, date: str) -> dict:
    """Return the forecast for one day at one location, or {"error"}."""
    input_error = check_in_israel(lat, lon) or _check_date(date)
    if input_error:
        return {"error": input_error}

    data, error = get_json(FORECAST_URL, {
        "latitude": lat, "longitude": lon, "daily": DAILY_FIELDS,
        "timezone": "Asia/Jerusalem", "start_date": date, "end_date": date,
    })
    if error:
        return {"error": error}

    try:
        day = {field: values[0] for field, values in data["daily"].items()}
    except (KeyError, IndexError, TypeError):
        return {"error": "The weather service returned no forecast for this date."}

    return {
        "date": date,
        "temp_max_c": day["temperature_2m_max"],
        "temp_min_c": day["temperature_2m_min"],
        "rain_chance_percent": day["precipitation_probability_max"],
        "rain_mm": day["precipitation_sum"],
        "wind_max_kmh": day["wind_speed_10m_max"],
        "uv_index_max": day["uv_index_max"],
        "sunrise": day["sunrise"][-5:],  # "2026-10-27T05:51" -> "05:51"
        "sunset": day["sunset"][-5:],
    }


def _check_date(text: str) -> str | None:
    """Return an error message if the date is invalid or outside the forecast range."""
    try:
        requested = datetime.date.fromisoformat(text)
    except (TypeError, ValueError):
        return f"'{text}' is not a valid date. Use the YYYY-MM-DD format."

    today = datetime.date.today()
    last_day = today + datetime.timedelta(days=MAX_DAYS_AHEAD)
    if requested < today:
        return f"{text} is in the past. The forecast starts today ({today})."
    if requested > last_day:
        return f"{text} is too far ahead. The forecast only reaches {last_day}."
    return None
