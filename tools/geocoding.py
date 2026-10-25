"""geocode tool: turns a place name into coordinates using OpenStreetMap (Nominatim).

The agent uses it for the starting city and to verify that places it suggests
really exist, since an LLM can invent or misspell place names.
"""
import time

from tools.http_utils import get_json

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MIN_SECONDS_BETWEEN_CALLS = 1  # Nominatim usage policy

_last_call_time = 0.0

GEOCODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "geocode",
        "description": "Find a place in Israel by name and return its coordinates. "
                       "Use it for the starting city and to verify every place you "
                       "want to suggest. Hebrew names work best.",
        "parameters": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "Place name, e.g. 'עין חמד'"},
            },
            "required": ["place"],
        },
    },
}


def geocode(place: str) -> dict:
    """Return {"name", "lat", "lon"} for a place in Israel, or {"error"}."""
    if not place or not place.strip():
        return {"error": "No place name was given."}

    _respect_rate_limit()
    results, error = get_json(NOMINATIM_URL, {
        "q": place, "format": "json", "limit": 1,
        "countrycodes": "il", "accept-language": "he",
    })
    if error:
        return {"error": error}
    if not results:
        return {"error": f"'{place}' was not found on the map. "
                         "Try the Hebrew name or a nearby well-known place."}

    best = results[0]
    short_name = ", ".join(best["display_name"].split(", ")[:3])
    return {"name": short_name,
            "lat": round(float(best["lat"]), 5),
            "lon": round(float(best["lon"]), 5)}


def _respect_rate_limit() -> None:
    """Sleep just long enough to keep at most one request per second."""
    global _last_call_time
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.time()
