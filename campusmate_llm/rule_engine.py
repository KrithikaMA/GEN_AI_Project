"""
rule_engine.py — Rule-Based Intent Detection & Response Engine
CampusMate Assistant | B.Tech Final Year Project
------------------------------------------------------
Handles deterministic, keyword-based query matching.
Uses NLP preprocessing (lowercase, punctuation removal, tokenization)
and maps user intent to structured responses from data.json.
"""

import json
import re
import random
import string
import os


class RuleEngine:
    """
    Rule-Based NLP Engine for campus query detection.
    Performs preprocessing and keyword-based intent matching.
    """

    def __init__(self, data_path: str = "data.json"):
        """Initialize engine and load knowledge base from data.json."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, data_path)

        with open(full_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.intents = self.data.get("intents", {})
        print(f"[RuleEngine] Loaded {len(self.intents)} intents from {data_path}")

    # ──────────────────────────────────────────────
    # NLP Preprocessing
    # ──────────────────────────────────────────────

    def preprocess(self, text: str) -> list:
        """
        NLP pipeline:
          1. Lowercase the text
          2. Remove punctuation
          3. Tokenize (split by whitespace)
        Returns a list of cleaned tokens.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        return tokens

    # ──────────────────────────────────────────────
    # Intent Detection
    # ──────────────────────────────────────────────

    def detect_intent(self, tokens: list) -> str | None:
        """
        Match tokens against keyword lists for each intent.
        Returns the best-matching intent name or None.
        Uses a scoring mechanism to pick the highest match.
        """
        scores = {}

        for intent_name, intent_data in self.intents.items():
            if intent_name == "unknown":
                continue  # Skip unknown; it's a fallback

            keywords = intent_data.get("keywords", [])
            score = 0

            for keyword in keywords:
                # Support multi-word keywords (e.g., "lab timing")
                keyword_tokens = keyword.split()
                if len(keyword_tokens) == 1:
                    if keyword in tokens:
                        score += 2  # Exact single-word match
                else:
                    # Check if all words of multi-word keyword appear in tokens
                    if all(kw in tokens for kw in keyword_tokens):
                        score += 3  # Multi-word match gets higher weight

            if score > 0:
                scores[intent_name] = score

        if not scores:
            return None

        # Return intent with highest score
        best_intent = max(scores, key=scores.get)
        return best_intent

    # ──────────────────────────────────────────────
    # Response Generation
    # ──────────────────────────────────────────────

    def get_response(self, intent: str | None) -> str | None:
        """
        Retrieve a random response for the detected intent.
        Returns None if no intent matched (triggers LLM fallback).
        """
        if intent is None:
            return None

        intent_data = self.intents.get(intent, {})
        responses = intent_data.get("responses", [])

        if not responses:
            return None

        return random.choice(responses)

    # ──────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────

    def process(self, user_message: str) -> str | None:
        """
        Full pipeline: preprocess → detect intent → return response.
        Returns None if no rule matched (LLM should handle it).
        """
        tokens = self.preprocess(user_message)
        intent = self.detect_intent(tokens)
        response = self.get_response(intent)

        if response:
            print(f"[RuleEngine] Intent: '{intent}' matched for: '{user_message}'")
        else:
            print(f"[RuleEngine] No match found for: '{user_message}' → forwarding to LLM")

        return response
