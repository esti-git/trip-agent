"""The agent loop: ask the LLM what to do next, run the tools it picks, repeat.

The loop ends when the LLM answers without asking for a tool (the goal is
reached, or it explains why it cannot be), or when MAX_STEPS is used up.
"""
import json

from openai import (APIConnectionError, APIStatusError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)
from openai.types.chat import ChatCompletionMessage

import config
from prompts import SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, run_tool

# Errors that mean "try again later or with another model", not "your request is wrong".
TEMPORARY_LLM_ERRORS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)


class AgentError(Exception):
    """Raised when the agent cannot talk to the LLM at all."""


class TripAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        fallbacks = [m.strip() for m in config.LLM_FALLBACK_MODELS.split(",") if m.strip()]
        self.models = [config.LLM_MODEL, *fallbacks]

    def run(self, goal: str) -> str:
        """Work toward the goal and return the agent's final answer."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": goal}]

        for step in range(1, config.MAX_STEPS + 1):
            reply = self._ask_llm(messages)
            messages.append(reply.model_dump(exclude_none=True))

            if not reply.tool_calls:
                return reply.content or "הסוכן סיים בלי לכתוב תשובה."

            if reply.content:
                print(f"💭 {reply.content.strip()}")
            for call in reply.tool_calls:
                result = run_tool(call.function.name, call.function.arguments)
                _print_step(step, call.function.name, call.function.arguments, result)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})

        return (f"לא הצלחתי להשלים תוכנית טיול תוך {config.MAX_STEPS} צעדים. "
                "נסו זמן נסיעה ארוך יותר או תאריך אחר.")

    def _ask_llm(self, messages: list) -> ChatCompletionMessage:
        """Send the conversation to the LLM, trying the fallback models if one is overloaded."""
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, tools=TOOL_SCHEMAS)
                self._prefer(model)
                return response.choices[0].message
            except TEMPORARY_LLM_ERRORS as e:
                last_error = e
                print(f"⚠️  המודל {model} לא זמין כרגע ({type(e).__name__})")
            except APIStatusError as e:
                raise AgentError(f"שירות ה-LLM דחה את הבקשה (קוד {e.status_code}). "
                                 "בדקו את LLM_API_KEY ואת LLM_MODEL בקובץ .env.") from e
        raise AgentError(f"שירות ה-LLM לא זמין כרגע ({type(last_error).__name__}). "
                         "נסו שוב בעוד כמה דקות.") from last_error

    def _prefer(self, model: str) -> None:
        """Try the model that just worked first next time, instead of the ones that failed."""
        self.models.remove(model)
        self.models.insert(0, model)


def _print_step(step: int, tool_name: str, arguments_json: str, result: dict) -> None:
    """Show one tool call and its result, so the user can follow the agent's reasoning."""
    try:
        arguments = ", ".join(f"{k}={v!r}" for k, v in json.loads(arguments_json).items())
    except (json.JSONDecodeError, AttributeError):
        arguments = arguments_json
    icon = "❌" if "error" in result else "✅"
    print(f"[{step}] 🔧 {tool_name}({arguments})")
    print(f"     {icon} {json.dumps(result, ensure_ascii=False)}")
