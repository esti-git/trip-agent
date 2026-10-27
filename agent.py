"""The agent loop: ask the LLM what to do next, run the tools it picks, repeat.

The loop ends when the LLM answers without asking for a tool (the goal is
reached, or it explains why it cannot be), or when MAX_STEPS is used up.

The agent does not print anything itself. It reports each step as an Event to
an on_event function, so the terminal (main.py) and the web UI (app.py) can
each show the steps in their own way.
"""
import json
from dataclasses import dataclass, field
from typing import Callable

from openai import (APIConnectionError, APIStatusError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)
from openai.types.chat import ChatCompletionMessage

import config
from prompts import SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, run_tool


class EmptyReplyError(Exception):
    """The LLM service answered, but with no reply in it."""


# Errors that mean "try again later or with another model", not "your request is wrong".
TEMPORARY_LLM_ERRORS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError,
                        EmptyReplyError)

# A final answer is a plan (its destination line) or an explanation of failure,
# as the system prompt requires. Any other text without a tool call is unfinished.
FINAL_ANSWER_MARKERS = ("📍", "❌")
MAX_NUDGES = 2
CONTINUE_NUDGE = ("Continue: call the next tool now. If you are done, give the final "
                  "answer in the required format.")

# Tools that take coordinates, and the argument pairs that hold them. These must be
# real places: the starting city or a geocode result, never numbers the LLM made up.
POINT_ARGUMENTS = {
    "get_weather": [("lat", "lon")],
    "travel_time": [("from_lat", "from_lon"), ("to_lat", "to_lon")],
}
COORDINATE_TOLERANCE = 0.001  # about 100 m: the LLM sometimes rounds the numbers
UNVERIFIED_POINT_ERROR = ("The coordinates ({lat}, {lon}) did not come from geocode or the "
                          "starting point. Call geocode for the place first and use the "
                          "exact coordinates it returns.")

# plan_schedule arguments that must repeat a value an earlier tool returned:
# argument -> (tool that returns it, key in that tool's result).
CHECKED_VALUES = {
    "sunset": ("get_weather", "sunset"),
    "drive_minutes": ("travel_time", "estimated_minutes"),
}
WRONG_LIMIT_ERROR = ("The user's maximum driving time is {limit} minutes, not {given}. "
                     "Call travel_time again with max_minutes={limit}.")
UNCHECKED_VALUE_ERROR = ("{argument}={given} is not a value {tool} returned (it returned: "
                         "{known}). Use the exact value from {tool}'s result.")


class AgentError(Exception):
    """Raised when the agent cannot talk to the LLM at all."""


@dataclass
class Event:
    """Something the agent did, reported so the user can follow its reasoning."""
    kind: str                  # "thought", "tool" or "warning"
    text: str = ""             # for "thought" and "warning"
    step: int = 0              # for "tool": which LLM turn asked for it
    tool: str = ""
    arguments: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)


class TripAgent:
    """Plans a day trip by letting the LLM choose tools until it gives a final answer."""

    def __init__(self, client: OpenAI, on_event: Callable[[Event], None] = lambda event: None):
        self.client = client
        self.on_event = on_event
        fallbacks = [m.strip() for m in config.LLM_FALLBACK_MODELS.split(",") if m.strip()]
        self.models = [config.LLM_MODEL, *fallbacks]

    def run(self, goal: str, verified_places: list[dict] = (),
            max_drive_minutes: int | None = None) -> str:
        """Work toward the goal and return the agent's final answer.

        verified_places are places already checked on the map (the starting city),
        as {"lat", "lon"} dicts. Places found by geocode during the run are added.
        max_drive_minutes is the user's limit; travel_time must be asked with it.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": goal}]
        self.verified_points = [(p["lat"], p["lon"]) for p in verified_places]
        self.max_drive_minutes = max_drive_minutes
        # Every value the tools returned for CHECKED_VALUES, e.g. {"sunset": {"16:52"}}.
        self.returned_values = {argument: set() for argument in CHECKED_VALUES}

        nudges = 0
        for step in range(1, config.MAX_STEPS + 1):
            # Planning needs data, so the first turn must use a tool rather than answer.
            reply = self._ask_llm(messages, must_use_tool=(step == 1))
            # Keep the whole reply, tool calls included: each tool result sent back
            # below must refer to the call it answers.
            messages.append(reply.model_dump(exclude_none=True))

            thought = clean_thought(reply.content or "")
            if not reply.tool_calls:
                if is_final_answer(reply.content or "") or nudges == MAX_NUDGES:
                    return reply.content or "הסוכן סיים בלי לכתוב תשובה."
                # Small models sometimes announce a check ("I will check...") without
                # making the tool call. That is not an answer, so ask it to go on.
                nudges += 1
                if thought:
                    self.on_event(Event("thought", text=thought))
                messages.append({"role": "user", "content": CONTINUE_NUDGE})
                continue

            if thought:
                self.on_event(Event("thought", text=thought))
            for call in reply.tool_calls:
                name, arguments = call.function.name, _parse_arguments(call.function.arguments)
                fact_error = (self._unverified_point_error(name, arguments) or
                              self._unverified_value_error(name, arguments))
                result = {"error": fact_error} if fact_error else run_tool(
                    name, call.function.arguments)
                if "error" not in result:
                    self._remember_facts(name, result)
                self.on_event(Event("tool", step=step, tool=name, arguments=arguments,
                                    result=result))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})

        return (f"לא הצלחתי להשלים תוכנית טיול תוך {config.MAX_STEPS} צעדים. "
                "נסו זמן נסיעה ארוך יותר או תאריך אחר.")

    def _ask_llm(self, messages: list, must_use_tool: bool = False) -> ChatCompletionMessage:
        """Send the conversation to the LLM, trying the fallback models if one is unavailable."""
        tool_choice = "required" if must_use_tool else "auto"
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice=tool_choice)
                if not response.choices:
                    raise EmptyReplyError(f"{model} returned no reply")
                self._prefer(model)
                return response.choices[0].message
            except TEMPORARY_LLM_ERRORS as e:
                last_error = e
                self.on_event(Event("warning", text=f"המודל {model} לא זמין כרגע ({type(e).__name__})"))
            except APIStatusError as e:
                raise AgentError(f"שירות ה-LLM דחה את הבקשה (קוד {e.status_code}). "
                                 "בדקו את LLM_API_KEY ואת LLM_MODEL בקובץ .env.") from e
        raise AgentError(f"שירות ה-LLM לא זמין כרגע ({type(last_error).__name__}). "
                         "נסו שוב בעוד כמה דקות.") from last_error

    def _unverified_point_error(self, tool: str, arguments: dict) -> str | None:
        """An error if the tool was given coordinates that are not a verified place.

        Without this, the LLM once could not find "פארק יקום", made up coordinates
        near the start, and got a 22-minute drive for a place an hour away.
        """
        for lat_key, lon_key in POINT_ARGUMENTS.get(tool, []):
            lat, lon = arguments.get(lat_key), arguments.get(lon_key)
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue  # missing or wrong types: the tool itself reports that
            if not any(abs(lat - v_lat) <= COORDINATE_TOLERANCE and
                       abs(lon - v_lon) <= COORDINATE_TOLERANCE
                       for v_lat, v_lon in self.verified_points):
                return UNVERIFIED_POINT_ERROR.format(lat=lat, lon=lon)
        return None

    def _unverified_value_error(self, tool: str, arguments: dict) -> str | None:
        """An error if the tool was given a number the user or an earlier tool did not give.

        Without this, the LLM could ask travel_time with a larger limit than the
        user's and get within_limit: true, or plan a schedule with a made-up sunset.
        """
        if tool == "travel_time" and self.max_drive_minutes is not None:
            given = arguments.get("max_minutes")
            if isinstance(given, (int, float)) and given != self.max_drive_minutes:
                return WRONG_LIMIT_ERROR.format(limit=self.max_drive_minutes, given=given)
        if tool == "plan_schedule":
            for argument, (source_tool, _) in CHECKED_VALUES.items():
                given, known = arguments.get(argument), self.returned_values[argument]
                if not isinstance(given, (str, int, float)):
                    continue  # missing or wrong types: the tool itself reports that
                if given not in known:
                    return UNCHECKED_VALUE_ERROR.format(
                        argument=argument, given=given, tool=source_tool,
                        known=", ".join(map(str, sorted(known))) or "nothing yet")
        return None

    def _remember_facts(self, tool: str, result: dict) -> None:
        """Keep what a tool returned, so later tool calls can be checked against it."""
        if tool == "geocode":
            self.verified_points.append((result["lat"], result["lon"]))
        for argument, (source_tool, key) in CHECKED_VALUES.items():
            if tool == source_tool and key in result:
                self.returned_values[argument].add(result[key])

    def _prefer(self, model: str) -> None:
        """Try the model that just worked first next time, instead of the ones that failed."""
        self.models.remove(model)
        self.models.insert(0, model)


def is_final_answer(text: str) -> bool:
    """True if the text is a finished answer: a plan (📍) or an explanation of failure (❌)."""
    return any(marker in text for marker in FINAL_ANSWER_MARKERS)


def clean_thought(text: str) -> str:
    """The LLM's explanation without lines of raw JSON.

    Small models sometimes repeat the tool arguments as JSON in their text,
    even when told not to. Those lines duplicate the tool call shown right
    after, so they are dropped here instead of relying on the prompt.
    """
    lines = [line for line in text.strip().splitlines()
             if not (line.strip().startswith("{") and line.strip().endswith("}"))]
    return "\n".join(lines).strip()


def _parse_arguments(arguments_json: str) -> dict:
    """Tool arguments as a dict for display; run_tool reports bad JSON to the LLM separately."""
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return {"raw": arguments_json}
    return arguments if isinstance(arguments, dict) else {"raw": arguments_json}
