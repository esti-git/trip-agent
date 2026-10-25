"""Shared HTTP helper for the tools: one place for timeouts and error handling."""
import requests

USER_AGENT = "trip-agent/1.0 (educational project)"
TIMEOUT_SECONDS = 15


def get_json(url: str, params: dict) -> tuple[dict | list | None, str | None]:
    """GET a JSON API.

    Returns (data, None) on success or (None, error message) on failure,
    so callers never have to catch network exceptions themselves.
    """
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS,
                                headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.json(), None
    except requests.Timeout:
        return None, "The service did not answer in time."
    except requests.RequestException as e:
        return None, f"The service is unavailable ({type(e).__name__})."
    except ValueError:
        return None, "The service returned an invalid response."
