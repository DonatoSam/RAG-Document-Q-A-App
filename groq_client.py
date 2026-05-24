"""
Groq API connectivity helper.

Note: The ChatGroq LLM class is now used directly from langchain_groq.
      This module only provides a standalone connection test utility.
"""

import os
import requests
from typing import Optional


def test_groq_connection(api_key: Optional[str] = None) -> tuple[bool, str]:
    """
    Ping the Groq API to verify connectivity and key validity.
    Returns (success: bool, message: str).
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        return False, "No API key provided."

    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=8,
        )
        if resp.status_code == 200:
            return True, "Connected to Groq API successfully."
        err_msg = resp.json().get("error", {}).get("message", resp.text)[:120]
        return False, f"HTTP {resp.status_code}: {err_msg}"

    except requests.exceptions.ConnectionError:
        return False, "Network error — check your internet connection."
    except requests.exceptions.Timeout:
        return False, "Connection timed out. Try again."
    except Exception as e:
        return False, f"Unexpected error: {e}"
