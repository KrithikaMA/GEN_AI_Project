"""
llm_engine.py — LLM Integration Engine (Ollama)
CampusMate Assistant | B.Tech Final Year Project
------------------------------------------------------
Connects to a locally running Ollama instance.
Sends campus-contextualised prompts and returns AI responses.
Includes domain restriction, context injection, and error handling.
"""

import requests
import json


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
REQUEST_TIMEOUT = 30  # seconds

SYSTEM_PROMPT = """You are CampusMate, a helpful and friendly academic assistant for a college campus.

Your ONLY role is to answer questions related to:
- Laboratory locations and timings
- Class and lecture schedules
- Examination schedules and dates
- Faculty information and contacts
- Campus facilities and general college information

STRICT RULES you must follow:
1. Only respond to campus-related academic queries.
2. If a user asks anything unrelated (politics, entertainment, sports, general coding help, etc.), 
   politely decline and redirect them to campus topics.
3. Keep all responses concise: 2–3 lines maximum unless a detailed list is genuinely required.
4. Always be friendly, professional, and encouraging.
5. If you don't know a specific fact, say so honestly rather than making something up.

Campus name: NIT Campus (National Institute of Technology)
"""


class LLMEngine:
    """
    LLM Engine that communicates with Ollama's local API.
    Injects conversation context and enforces domain restrictions.
    """

    def __init__(
        self,
        url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = REQUEST_TIMEOUT
    ):
        self.url = url
        self.model = model
        self.timeout = timeout
        print(f"[LLMEngine] Configured → model: {model} | endpoint: {url}")

    # ──────────────────────────────────────────────
    # Prompt Construction
    # ──────────────────────────────────────────────

    def build_prompt(self, user_message: str, context_string: str = "") -> str:
        """
        Build the full prompt to send to the LLM.
        Includes system instructions, conversation history, and new query.
        """
        prompt_parts = [SYSTEM_PROMPT]

        if context_string:
            prompt_parts.append(
                f"\n--- Previous Conversation ---\n{context_string}\n--- End of History ---\n"
            )

        prompt_parts.append(f"\nUser's current question: {user_message}\n\nCampusMate:")

        return "\n".join(prompt_parts)

    # ──────────────────────────────────────────────
    # API Call
    # ──────────────────────────────────────────────

    def query(self, user_message: str, context_string: str = "") -> str:
        """
        Send a request to Ollama and return the response text.
        Handles connection errors, timeouts, and malformed responses.
        """
        prompt = self.build_prompt(user_message, context_string)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get full response at once
            "options": {
                "temperature": 0.7,     # Balanced creativity
                "top_p": 0.9,
                "num_predict": 256,     # Keep responses concise
            }
        }

        print(f"[LLMEngine] Sending query to Ollama: '{user_message[:60]}...'")

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()

            data = response.json()
            llm_response = data.get("response", "").strip()

            if not llm_response:
                return self._fallback_response("empty_response")

            print(f"[LLMEngine] Response received ({len(llm_response)} chars)")
            return llm_response

        except requests.exceptions.ConnectionError:
            print("[LLMEngine] ERROR: Cannot connect to Ollama. Is it running?")
            return self._fallback_response("connection_error")

        except requests.exceptions.Timeout:
            print("[LLMEngine] ERROR: Ollama request timed out.")
            return self._fallback_response("timeout")

        except requests.exceptions.HTTPError as e:
            print(f"[LLMEngine] HTTP Error: {e}")
            return self._fallback_response("http_error")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[LLMEngine] Response parsing error: {e}")
            return self._fallback_response("parse_error")

        except Exception as e:
            print(f"[LLMEngine] Unexpected error: {e}")
            return self._fallback_response("unknown_error")

    # ──────────────────────────────────────────────
    # Fallback Messages
    # ──────────────────────────────────────────────

    def _fallback_response(self, error_type: str) -> str:
        """Return a user-friendly error message based on error type."""
        messages = {
            "connection_error": (
                "⚠️ I'm having trouble connecting to my AI brain right now. "
                "Please make sure Ollama is running (`ollama serve`), "
                "or ask a simpler question I can answer directly!"
            ),
            "timeout": (
                "⏳ That took too long to process. "
                "Try rephrasing your question more concisely!"
            ),
            "empty_response": (
                "🤔 I processed your question but couldn't form a response. "
                "Could you try rephrasing it?"
            ),
            "http_error": (
                "⚠️ There was a server error with the AI model. "
                "Please try again in a moment."
            ),
            "parse_error": (
                "⚠️ I received an unexpected response format. "
                "Please try again."
            ),
            "unknown_error": (
                "⚠️ Something unexpected went wrong. "
                "Please try again or ask a different question."
            ),
        }
        return messages.get(error_type, messages["unknown_error"])

    # ──────────────────────────────────────────────
    # Health Check
    # ──────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            resp = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                available = [m.get("name", "") for m in models]
                print(f"[LLMEngine] Ollama healthy. Models: {available}")
                return True
        except Exception:
            pass
        print("[LLMEngine] Ollama health check failed.")
        return False
