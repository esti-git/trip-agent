"""travel_time and plan_schedule tools: pure calculations, no internet needed.

travel_time estimates the drive from the straight-line distance, since a
free routing service was not reliable enough. plan_schedule checks that a
visit fits in the day and that the trip ends before sunset.
"""
import datetime
import math

from tools.geocoding import check_in_israel

EARTH_RADIUS_KM = 6371
ROAD_FACTOR = 1.3          # roads are about 30% longer than a straight line
AVERAGE_SPEED_KMH = 60
SUNSET_BUFFER_MINUTES = 30  # leave the place at least this long before sunset

TRAVEL_TIME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "travel_time",
        "description": "Estimate the driving distance and time between two points, "
                       "and whether it is within the user's maximum driving time.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_lat": {"type": "number"},
                "from_lon": {"type": "number"},
                "to_lat": {"type": "number"},
                "to_lon": {"type": "number"},
                "max_minutes": {"type": "integer",
                                "description": "The user's maximum one-way driving time"},
            },
            "required": ["from_lat", "from_lon", "to_lat", "to_lon", "max_minutes"],
        },
    },
}

PLAN_SCHEDULE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "plan_schedule",
        "description": "Build a day schedule for a trip and check that the visit ends "
                       f"at least {SUNSET_BUFFER_MINUTES} minutes before sunset. "
                       "If it does not fit, the result says by how much.",
        "parameters": {
            "type": "object",
            "properties": {
                "leave_home": {"type": "string", "description": "Departure time, HH:MM"},
                "drive_minutes": {"type": "integer", "description": "One-way drive time"},
                "visit_hours": {"type": "number", "description": "Time to spend at the place"},
                "sunset": {"type": "string", "description": "Sunset time, HH:MM"},
            },
            "required": ["leave_home", "drive_minutes", "visit_hours", "sunset"],
        },
    },
}


def travel_time(from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                max_minutes: int) -> dict:
    """Estimate the drive between two points and compare it to the user's limit."""
    for lat, lon in ((from_lat, from_lon), (to_lat, to_lon)):
        location_error = check_in_israel(lat, lon)
        if location_error:
            return {"error": location_error}
    if max_minutes <= 0:
        return {"error": "The maximum driving time must be positive."}

    straight_km = _haversine_km(from_lat, from_lon, to_lat, to_lon)
    road_km = straight_km * ROAD_FACTOR
    minutes = round(road_km / AVERAGE_SPEED_KMH * 60)
    return {
        "straight_line_km": round(straight_km, 1),
        "estimated_road_km": round(road_km, 1),
        "estimated_minutes": minutes,
        "within_limit": minutes <= max_minutes,
    }


def plan_schedule(leave_home: str, drive_minutes: int, visit_hours: float,
                  sunset: str) -> dict:
    """Lay out the trip's times and check it finishes before sunset."""
    try:
        start = _parse_time(leave_home)
        sunset_time = _parse_time(sunset)
    except ValueError as e:
        return {"error": str(e)}
    if drive_minutes < 0 or visit_hours <= 0:
        return {"error": "Drive time cannot be negative and visit time must be positive."}

    arrive = start + datetime.timedelta(minutes=drive_minutes)
    leave_place = arrive + datetime.timedelta(hours=visit_hours)
    back_home = leave_place + datetime.timedelta(minutes=drive_minutes)
    latest_leave_place = sunset_time - datetime.timedelta(minutes=SUNSET_BUFFER_MINUTES)
    spare_minutes = round((latest_leave_place - leave_place).total_seconds() / 60)

    return {
        "leave_home": _format_time(start),
        "arrive": _format_time(arrive),
        "leave_place": _format_time(leave_place),
        "back_home": _format_time(back_home),
        "latest_leave_place": _format_time(latest_leave_place),
        "fits_before_sunset": spare_minutes >= 0,
        "spare_minutes": spare_minutes,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance between two points on Earth, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _parse_time(text: str) -> datetime.datetime:
    """Parse "HH:MM" into a datetime on a fixed day, so times can be added and compared."""
    try:
        return datetime.datetime.strptime(text, "%H:%M")
    except (TypeError, ValueError):
        raise ValueError(f"'{text}' is not a valid time. Use the HH:MM format.") from None


def _format_time(moment: datetime.datetime) -> str:
    """Format a time as "HH:MM", marking times that roll past midnight."""
    suffix = " (next day)" if moment.day > 1 else ""
    return moment.strftime("%H:%M") + suffix
