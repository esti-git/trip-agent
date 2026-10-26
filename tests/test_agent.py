"""Tests for cleaning the agent's explanations. No LLM or internet needed."""
from agent import clean_thought


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
