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

# Errors that mean "try again later or with another model", not "your request is wrong".
TEMPORARY_LLM_ERRORS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)

# A final answer is a plan (its destination line) or an explanation of failure,
# as the system prompt requires. Any other text without a tool call is unfinished.
FINAL_ANSWER_MARKERS = ("📍", "❌")
MAX_NUDGES = 2
CONTINUE_NUDGE = ("Continue: call the next tool now. If you are done, give the final "
                  "answer in the required format.")


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
    def __init__(self, client: OpenAI, on_event: Callable[[Event], None] = lambda event: None):
        self.client = client
        self.on_event = on_event
        fallbacks = [m.strip() for m in config.LLM_FALLBACK_MODELS.split(",") if m.strip()]
        self.models = [config.LLM_MODEL, *fallbacks]

    def run(self, goal: str) -> str:
        """Work toward the goal and return the agent's final answer."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": goal}]

        nudges = 0
        for step in range(1, config.MAX_STEPS + 1):
            # Planning needs data, so the first turn must use a tool rather than answer.
            reply = self._ask_llm(messages, must_use_tool=(step == 1))
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
                result = run_tool(call.function.name, call.function.arguments)
                self.on_event(Event("tool", step=step, tool=call.function.name,
                                    arguments=_parse_arguments(call.function.arguments),
                                    result=result))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})

        return (f"לא הצלחתי להשלים תוכנית טיול תוך {config.MAX_STEPS} צעדים. "
                "נסו זמן נסיעה ארוך יותר או תאריך אחר.")

    def _ask_llm(self, messages: list, must_use_tool: bool = False) -> ChatCompletionMessage:
        """Send the conversation to the LLM, trying the fallback models if one is overloaded."""
        tool_choice = "required" if must_use_tool else "auto"
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice=tool_choice)
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
