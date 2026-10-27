"""geocode tool: turns a place name into coordinates using OpenStreetMap (Nominatim).

The agent uses it to verify that places it suggests really exist, since an LLM
can invent or misspell place names. find_settlement checks the user's starting
city before the agent runs.
"""
import time

from tools.http_utils import get_json

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MIN_SECONDS_BETWEEN_CALLS = 1  # Nominatim usage policy
# Many trip sites (Qumran, Herodion, Kfar Etzion...) are mapped under "ps" in
# OpenStreetMap. Searching "il" alone returns same-named streets instead.
COUNTRY_CODES = "il,ps"
# Several results, not just the best one: the real place is often listed after a
# same-named street (see _prefer_non_street and pick_settlement).
CANDIDATES = 5
# OpenStreetMap classes of cities, towns, villages and neighborhoods. Everything
# else (a street, a yeshiva, an event hall) is not a valid starting city.
SETTLEMENT_CLASSES = {"place", "boundary"}

# A box around Israel, from Eilat to the Golan. Used by the other tools to catch
# coordinates the LLM mixed up, such as latitude and longitude swapped.
ISRAEL_LAT_RANGE = (29.3, 33.5)
ISRAEL_LON_RANGE = (34.2, 36.0)

_last_call_time = 0.0

GEOCODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "geocode",
        "description": "Find a place in Israel by name and return its coordinates. "
                       "Use it to verify every place you want to suggest; other tools "
                       "accept only coordinates it returned. Hebrew names work best.",
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

    results, error = _search(place)
    if error:
        return {"error": error}
    if not results:
        return {"error": f"'{place}' was not found on the map. "
                         "Try the Hebrew name or a nearby well-known place."}
    return _to_place(_prefer_non_street(results))


def find_settlement(name: str) -> dict:
    """Find the user's starting city: a city, town, village or neighborhood.

    Returns {"name", "lat", "lon"}, or {"error"} with a Hebrew message for the
    user (unlike geocode, whose errors are read by the LLM). If the name only
    matches something else, such as "ישיבת מיר מודיעין עלית" for the misspelled
    "מודיעין עלית", the message says what was found so the user can fix the spelling.
    """
    if not name or not name.strip():
        return {"error": "יש להזין שם של עיר."}

    results, error = _search(name)
    if error:
        return {"error": "שירות המפות לא זמין כרגע. נסו שוב בעוד רגע."}
    if not results:
        return {"error": f"לא מצאתי את '{name}' על המפה. בדקו את הכתיב ונסו שוב."}

    settlement = pick_settlement(results)
    if settlement is None:
        closest = _to_place(results[0])["name"]
        return {"error": f"לא מצאתי יישוב בשם '{name}'. המקום הכי קרוב שמצאתי: {closest}. "
                         "נסו כתיב אחר, למשל עם או בלי י'."}
    return _to_place(settlement)


def pick_settlement(results: list[dict]) -> dict | None:
    """The first search result that is a settlement, or None if there is none."""
    return next((r for r in results if r.get("class") in SETTLEMENT_CLASSES), None)


def _search(query: str) -> tuple[list | None, str | None]:
    """Ask Nominatim for up to CANDIDATES matches. Returns (results, error)."""
    _respect_rate_limit()
    results, error = get_json(NOMINATIM_URL, {
        "q": query, "format": "json", "limit": CANDIDATES,
        "countrycodes": COUNTRY_CODES, "accept-language": "he",
    })
    if error:
        return None, error
    # When overloaded, the service can answer with an object such as {"error": "..."}
    # instead of a list of places. Keep only well-formed results.
    if not isinstance(results, list):
        return None, "The map service returned an unexpected response."
    return [r for r in results
            if isinstance(r, dict) and {"lat", "lon", "display_name"} <= r.keys()], None


def _to_place(result: dict) -> dict:
    """Keep only what the agent needs from a search result: a short name and coordinates."""
    return {"name": ", ".join(result["display_name"].split(", ")[:3]),
            "lat": round(float(result["lat"]), 5),
            "lon": round(float(result["lon"]), 5)}


def _prefer_non_street(results: list[dict]) -> dict:
    """Pick the first result that is not a street or bus stop named after the place.

    Searching "כפר עציון" otherwise finds Kfar Etzion Street in Jerusalem before
    the village itself. If only streets match, the first one is still a fair guess.
    """
    return next((r for r in results if r.get("class") != "highway"), results[0])


def check_in_israel(lat: float, lon: float) -> str | None:
    """Return an error message if the point is outside Israel, otherwise None."""
    if (ISRAEL_LAT_RANGE[0] <= lat <= ISRAEL_LAT_RANGE[1]
            and ISRAEL_LON_RANGE[0] <= lon <= ISRAEL_LON_RANGE[1]):
        return None
    return (f"The point ({lat}, {lon}) is outside Israel. "
            "Use the exact coordinates returned by geocode.")


def _respect_rate_limit() -> None:
    """Sleep just long enough to keep at most one request per second."""
    global _last_call_time
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.time()
