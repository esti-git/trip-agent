"""Tests for choosing the starting city from map search results. No internet needed.

The results below have the shape Nominatim returns, trimmed to the fields used.
"""
from tools.geocoding import pick_settlement

CITY = {"class": "boundary", "type": "administrative", "display_name": "מודיעין עילית"}
VILLAGE = {"class": "place", "type": "village", "display_name": "יגור"}
NEIGHBORHOOD = {"class": "place", "type": "suburb", "display_name": "רמת אביב"}
YESHIVA = {"class": "amenity", "type": "place_of_worship", "display_name": "ישיבת מיר מודיעין עלית"}
STREET = {"class": "highway", "type": "residential", "display_name": "מודיעין, טבריה"}


def test_city_is_a_settlement():
    assert pick_settlement([CITY]) is CITY


def test_village_and_neighborhood_are_settlements():
    assert pick_settlement([VILLAGE]) is VILLAGE
    assert pick_settlement([NEIGHBORHOOD]) is NEIGHBORHOOD


def test_settlement_is_preferred_over_earlier_non_settlements():
    assert pick_settlement([YESHIVA, STREET, CITY]) is CITY


def test_only_non_settlements_means_none():
    # The real case: "מודיעין עלית" (misspelled) matched only a yeshiva and an event hall.
    assert pick_settlement([YESHIVA, STREET]) is None


def test_no_results_means_none():
    assert pick_settlement([]) is None
