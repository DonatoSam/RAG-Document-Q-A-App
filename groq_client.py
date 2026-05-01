"""Minimal Groq LLM client implementing LangChain LLM interface.

This wrapper implements the bare-minimum _call method so it can be
used with LangChain's high-level chains (RetrievalQA, etc.).

Adjust api_base or payload if Groq changes their API surface.
"""

from langchain_core.language_models.llms import LLM
from typing import Optional, Mapping, Any
import requests
import os

def test_groq_connection(api_key: Optional[str] = None) -> bool:
    """Ping the Groq API to test connectivity and key validity."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return False
        
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {key}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
        
class ChatGroq(LLM):
    """Simple Groq LLM wrapper."""
    model: str = "llama3-8b-8192"
    api_key: Optional[str] = None
    api_base: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def _call(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> str:
        key = self.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY missing - set it in the environment or pass api_key")

        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.temperature),
            "max_tokens": 1024,
        }

        retries = 3
        timeout_seconds = 15
        
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                try:
                    return data.get("choices", [])[0].get("message", {}).get("content", "")
                except Exception:
                    return f"Error: Unexpected response format from API: {data}"
            except requests.exceptions.Timeout:
                if attempt == retries - 1:
                    return "Error: The Groq API connection timed out. Please try again later."
            except requests.exceptions.ConnectionError as e:
                if attempt == retries - 1:
                    return f"Error: Could not connect to the Groq API. Please check your network connection or API URL. Details: {str(e)}"
            except requests.exceptions.RequestException as e:
                return f"Error: API request failed: {str(e)}"
            except Exception as e:
                return f"Error: An unexpected error occurred: {str(e)}"
        
        return "Error: Max retries exceeded when calling Groq API."
