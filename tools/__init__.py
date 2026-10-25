"""The agent's toolbox: the list of tools the LLM may call, and one function to run them.

Each tool lives in its own module next to its schema (the description the LLM
reads). To add a tool, write its module and register it in TOOLS below.
"""
import json

from tools.geocoding import GEOCODE_SCHEMA, geocode
from tools.travel import PLAN_SCHEDULE_SCHEMA, TRAVEL_TIME_SCHEMA, plan_schedule, travel_time
from tools.weather import GET_WEATHER_SCHEMA, get_weather

TOOLS = {
    "geocode": (geocode, GEOCODE_SCHEMA),
    "get_weather": (get_weather, GET_WEATHER_SCHEMA),
    "travel_time": (travel_time, TRAVEL_TIME_SCHEMA),
    "plan_schedule": (plan_schedule, PLAN_SCHEDULE_SCHEMA),
}

TOOL_SCHEMAS = [schema for _, schema in TOOLS.values()]


def run_tool(name: str, arguments_json: str) -> dict:
    """Run the tool the LLM asked for. Never raises: problems come back as {"error"}."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."}

    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return {"error": f"The arguments for {name} were not valid JSON."}

    function, _ = TOOLS[name]
    try:
        return function(**arguments)
    except TypeError as e:
        return {"error": f"Wrong arguments for {name}: {e}"}
    except Exception as e:  # last line of defense: one broken tool call must not crash the agent
        return {"error": f"{name} failed unexpectedly ({type(e).__name__}: {e})."}
