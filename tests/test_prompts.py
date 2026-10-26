"""Tests for the goal message given to the agent. No LLM or internet needed."""
import datetime

import pytest

from prompts import TRIP_TYPES, build_goal

ORIGIN = {"name": "ירושלים", "lat": 31.77885, "lon": 35.22579}
TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()


def test_goal_contains_the_users_input():
    goal = build_goal(ORIGIN, TOMORROW, 60)
    assert "ירושלים" in goal
    assert "31.77885" in goal and "35.22579" in goal
    assert TOMORROW in goal
    assert "60 minutes" in goal


@pytest.mark.parametrize("trip_type", list(TRIP_TYPES))
def test_preferred_trip_type_is_passed_to_the_agent(trip_type):
    goal = build_goal(ORIGIN, TOMORROW, 60, trip_type)
    assert f"Preferred kind of trip: {TRIP_TYPES[trip_type][1]}" in goal


def test_no_preference_lets_the_agent_choose():
    goal = build_goal(ORIGIN, TOMORROW, 60, None)
    assert "Preferred kind of trip" not in goal
    assert "No preferred kind of trip" in goal


def test_trip_today_includes_the_current_time():
    goal = build_goal(ORIGIN, datetime.date.today().isoformat(), 60)
    assert "it is now" in goal


def test_trip_on_another_day_has_no_current_time():
    assert "it is now" not in build_goal(ORIGIN, TOMORROW, 60)
