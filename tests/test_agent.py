"""Tests for the agent: cleaning its explanations and running its loop.

The loop tests use a fake LLM with scripted replies, so no LLM or internet is needed.
"""
import json
from types import SimpleNamespace

from openai.types.chat import ChatCompletionMessage

from agent import TripAgent, clean_thought, is_final_answer


def test_plain_sentence_is_kept():
    assert clean_thought("אני בודק את מזג האוויר בטבריה.") == "אני בודק את מזג האוויר בטבריה."


def test_json_line_is_removed():
    text = 'אני בודק את זמן הנסיעה.\n\n{"from_lat":32.79,"to_lat":32.85,"max_minutes":60}'
    assert clean_thought(text) == "אני בודק את זמן הנסיעה."


def test_indented_json_line_is_removed():
    assert clean_thought('בודק.\n   {"place": "עין חמד"}   ') == "בודק."


def test_only_json_leaves_nothing():
    assert clean_thought('{"place": "עין חמד"}') == ""


def test_empty_text_stays_empty():
    assert clean_thought("") == ""


def test_text_with_braces_inside_a_sentence_is_kept():
    text = "הכלי מחזיר {temp} ו-{rain} לכל יום."
    assert clean_thought(text) == text


# --- the agent loop, with a fake LLM that returns scripted replies ----------

PLAN = "🌤️ תחזית: נעים\n🎯 סוג הטיול: טבע\n📍 יעד: עין חמד (10 ק\"מ)\n🕗 לוח זמנים: ...\n💡 טיפ: מים"
TRAVEL_CALL = {"id": "call_1", "type": "function", "function": {
    "name": "travel_time",
    "arguments": json.dumps({"from_lat": 31.78, "from_lon": 35.22, "to_lat": 31.79,
                             "to_lon": 35.14, "max_minutes": 60})}}


def reply(content=None, tool_calls=None):
    return ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": content, "tool_calls": tool_calls})


class FakeLLM:
    """Stands in for the OpenAI client: returns the scripted replies and records each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


def test_first_turn_must_use_a_tool_and_later_turns_may_answer():
    llm = FakeLLM(reply(tool_calls=[TRAVEL_CALL]), reply(PLAN))
    assert TripAgent(llm).run("goal") == PLAN
    assert llm.requests[0]["tool_choice"] == "required"
    assert llm.requests[1]["tool_choice"] == "auto"


def test_announcing_a_check_without_a_tool_call_is_not_the_answer():
    # The real bug: the model wrote "I am checking the weather" and called nothing.
    llm = FakeLLM(reply(tool_calls=[TRAVEL_CALL]), reply("אני בודק את מזג האוויר."), reply(PLAN))
    events = []
    assert TripAgent(llm, on_event=events.append).run("goal") == PLAN
    nudges = [m for m in llm.requests[-1]["messages"]
              if m.get("role") == "user" and "Continue" in m["content"]]
    assert len(nudges) == 1
    assert any(e.kind == "thought" and "בודק" in e.text for e in events)


def test_explanation_of_failure_is_a_final_answer():
    failure = "❌ לא מצאתי טיול שמסתיים לפני השקיעה."
    llm = FakeLLM(reply(tool_calls=[TRAVEL_CALL]), reply(failure))
    assert TripAgent(llm).run("goal") == failure


def test_gives_up_nudging_after_two_tries():
    llm = FakeLLM(reply(tool_calls=[TRAVEL_CALL]), reply("רגע"), reply("עוד רגע"), reply("סיימתי"))
    assert TripAgent(llm).run("goal") == "סיימתי"
    assert len(llm.requests) == 4


def test_tool_results_are_reported_as_events():
    events = []
    TripAgent(FakeLLM(reply(tool_calls=[TRAVEL_CALL]), reply(PLAN)), on_event=events.append).run("goal")
    tool_events = [e for e in events if e.kind == "tool"]
    assert tool_events[0].tool == "travel_time"
    assert tool_events[0].result["within_limit"] is True


def test_final_answer_markers():
    assert is_final_answer(PLAN)
    assert is_final_answer("❌ לא הצלחתי")
    assert not is_final_answer("אני בודק את מזג האוויר.")
