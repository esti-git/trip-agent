"""Tests for the travel calculations. Run from the project root: python -m pytest"""
import pytest

from tools.travel import plan_schedule, travel_time


# --- travel_time -----------------------------------------------------------

def test_same_point_is_zero_distance():
    result = travel_time(31.78, 35.22, 31.78, 35.22, max_minutes=60)
    assert result["straight_line_km"] == 0
    assert result["estimated_minutes"] == 0
    assert result["within_limit"] is True


def test_one_degree_of_latitude_is_about_111_km():
    result = travel_time(31.0, 35.0, 32.0, 35.0, max_minutes=300)
    assert result["straight_line_km"] == pytest.approx(111.2, abs=0.5)


def test_road_distance_is_30_percent_longer_and_driven_at_60_kmh():
    result = travel_time(31.0, 35.0, 32.0, 35.0, max_minutes=300)
    assert result["estimated_road_km"] == pytest.approx(111.2 * 1.3, abs=1)
    # At 60 km/h, minutes equal kilometers.
    assert result["estimated_minutes"] == pytest.approx(result["estimated_road_km"], abs=1)


def test_jerusalem_to_tel_aviv():
    result = travel_time(31.7789, 35.2258, 32.0853, 34.7818, max_minutes=90)
    assert 50 <= result["straight_line_km"] <= 58
    assert result["within_limit"] is True


def test_too_far_is_outside_the_limit():
    result = travel_time(31.0, 35.0, 32.0, 35.0, max_minutes=60)
    assert result["within_limit"] is False


@pytest.mark.parametrize("coords", [
    (95, 35, 31, 35),          # not a valid latitude
    (31, 35, 31, 200),         # not a valid longitude
    (31.25, 34.79, 30.79, 30.79),  # lat and lon mixed up: a point in Egypt
])
def test_points_outside_israel_return_error(coords):
    assert "error" in travel_time(*coords, max_minutes=60)


def test_non_positive_limit_returns_error():
    assert "error" in travel_time(31.0, 35.0, 32.0, 35.0, max_minutes=0)


# --- plan_schedule ---------------------------------------------------------

def test_schedule_that_fits():
    result = plan_schedule("08:00", drive_minutes=45, visit_hours=3, sunset="17:00")
    assert result["arrive"] == "08:45"
    assert result["leave_place"] == "11:45"
    assert result["back_home"] == "12:30"
    assert result["latest_leave_place"] == "16:30"
    assert result["fits_before_sunset"] is True
    assert result["spare_minutes"] == 285


def test_schedule_that_ends_too_late():
    result = plan_schedule("13:00", drive_minutes=60, visit_hours=4, sunset="17:00")
    assert result["leave_place"] == "18:00"
    assert result["fits_before_sunset"] is False
    assert result["spare_minutes"] == -90


def test_leaving_exactly_at_the_latest_time_still_fits():
    result = plan_schedule("12:00", drive_minutes=30, visit_hours=4, sunset="17:00")
    assert result["leave_place"] == "16:30"
    assert result["fits_before_sunset"] is True
    assert result["spare_minutes"] == 0


def test_trip_past_midnight_is_marked_next_day():
    result = plan_schedule("22:00", drive_minutes=60, visit_hours=2, sunset="17:00")
    assert result["back_home"] == "02:00 (next day)"
    assert result["fits_before_sunset"] is False


@pytest.mark.parametrize("leave_home, sunset", [("8am", "17:00"), ("08:00", "25:00"), (None, "17:00")])
def test_bad_time_format_returns_error(leave_home, sunset):
    assert "error" in plan_schedule(leave_home, drive_minutes=30, visit_hours=2, sunset=sunset)


@pytest.mark.parametrize("drive_minutes, visit_hours", [(-10, 2), (30, 0)])
def test_invalid_durations_return_error(drive_minutes, visit_hours):
    assert "error" in plan_schedule("08:00", drive_minutes, visit_hours, sunset="17:00")
