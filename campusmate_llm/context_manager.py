"""
context_manager.py — Conversation Memory Manager
CampusMate Assistant | B.Tech Final Year Project
------------------------------------------------------
Maintains a rolling window of the last N conversation turns.
Formats conversation history for injection into LLM prompts.
Each "turn" = one user message + one bot response.
"""


class ContextManager:
    """
    Manages short-term conversation memory.
    Stores last `max_turns` of (user, bot) pairs.
    """

    def __init__(self, max_turns: int = 3):
        """
        Initialize the context window.
        :param max_turns: Number of past turns to retain (default: 3)
        """
        self.max_turns = max_turns
        self.history = []  # List of dicts: {"user": ..., "bot": ...}
        print(f"[ContextManager] Initialized with max_turns={max_turns}")

    def add_turn(self, user_message: str, bot_response: str):
        """
        Add a new conversation turn to history.
        Automatically drops oldest turn if limit exceeded.
        """
        self.history.append({
            "user": user_message.strip(),
            "bot": bot_response.strip()
        })

        # Maintain rolling window
        if len(self.history) > self.max_turns:
            self.history.pop(0)

        print(f"[ContextManager] History size: {len(self.history)}/{self.max_turns}")

    def get_context_string(self) -> str:
        """
        Format conversation history as a readable string
        for injection into the LLM system prompt.
        Returns empty string if no history exists.
        """
        if not self.history:
            return ""

        lines = []
        for turn in self.history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot: {turn['bot']}")

        return "\n".join(lines)

    def get_history(self) -> list:
        """Return raw history list (for API/debug use)."""
        return self.history.copy()

    def clear(self):
        """Reset conversation history."""
        self.history = []
        print("[ContextManager] History cleared.")

    def __len__(self):
        return len(self.history)
