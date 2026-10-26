"""Settings and the LLM client.

The LLM is reached through the `openai` library. By default it talks to
Gemini's OpenAI-compatible API (free); switching to OpenAI or another
provider only needs LLM_BASE_URL and LLM_MODEL in .env, not a code change.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
# Tried in order when the main model is overloaded. Comma-separated; empty disables fallback.
LLM_FALLBACK_MODELS = os.getenv("LLM_FALLBACK_MODELS", "gemini-flash-latest,gemini-flash-lite-latest")

MIN_DRIVE_MINUTES = 10    # limits for the user's maximum one-way drive
MAX_DRIVE_MINUTES = 300

MAX_STEPS = 15            # the agent gives up after this many LLM turns
# Kept short on purpose: a stuck model should hand over to the fallback quickly.
LLM_MAX_RETRIES = 1       # retries on temporary errors such as "model overloaded"
LLM_TIMEOUT_SECONDS = 30


class ConfigError(Exception):
    """Raised when the agent cannot start because of missing settings."""


def create_llm_client() -> OpenAI:
    """Build the LLM client, or raise ConfigError with a message the user can act on."""
    if not LLM_API_KEY:
        raise ConfigError("חסר מפתח LLM_API_KEY. העתיקו את .env.example לקובץ .env והוסיפו בו את המפתח.")
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                  max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT_SECONDS)
