"""
app.py — Flask Backend Server
CampusMate Assistant | B.Tech Final Year Project
------------------------------------------------------
Main entry point for the CampusMate web application.
Orchestrates the Hybrid Engine:
    1. Rule Engine  (fast, deterministic)
    2. LLM Engine   (Ollama fallback)
    3. Context Manager (conversation memory)

Routes:
    GET  /          → Serve chat UI
    POST /chat      → Process user message
    GET  /health    → System status check
    POST /reset     → Clear conversation history
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from rule_engine import RuleEngine
from llm_engine import LLMEngine
from context_manager import ContextManager

# ──────────────────────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (useful for development)

# Initialize all modules
rule_engine = RuleEngine(data_path="data.json")
llm_engine = LLMEngine()
context_manager = ContextManager(max_turns=3)

print("=" * 55)
print("  CampusMate Assistant — Hybrid LLM Backend Ready")
print("  http://localhost:5000")
print("=" * 55)


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    
    Request body (JSON):
        { "message": "Where is the CS lab?" }

    Response (JSON):
        { "response": "...", "source": "rule" | "llm" }
    """
    # ── Input Validation ──────────────────────────
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "response": "⚠️ Invalid request format. Please send a JSON body.",
            "source": "error"
        }), 400

    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "response": "Please type a message before sending! 💬",
            "source": "error"
        }), 400

    if len(user_message) > 500:
        return jsonify({
            "response": "Message too long. Please keep it under 500 characters.",
            "source": "error"
        }), 400

    # ── Hybrid Engine Logic ────────────────────────
    response = None
    source = "rule"

    # Step 1: Try Rule Engine first (fast path)
    rule_response = rule_engine.process(user_message)

    if rule_response:
        response = rule_response
        source = "rule"
    else:
        # Step 2: Fallback to LLM Engine
        context_string = context_manager.get_context_string()
        llm_response = llm_engine.query(user_message, context_string)
        response = llm_response
        source = "llm"

    # ── Update Context Memory ──────────────────────
    context_manager.add_turn(user_message, response)

    # ── Return Response ────────────────────────────
    return jsonify({
        "response": response,
        "source": source
    })


@app.route("/health", methods=["GET"])
def health():
    """
    System health check endpoint.
    Returns status of Rule Engine, LLM Engine, and Context Manager.
    """
    ollama_status = llm_engine.health_check()

    return jsonify({
        "status": "ok",
        "components": {
            "rule_engine": "active",
            "llm_engine": "active" if ollama_status else "unreachable",
            "context_manager": f"active ({len(context_manager)} turns stored)",
            "ollama_connected": ollama_status
        }
    })


@app.route("/reset", methods=["POST"])
def reset_context():
    """Clear the conversation history (start fresh session)."""
    context_manager.clear()
    return jsonify({
        "status": "ok",
        "message": "Conversation history cleared."
    })


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=True
    )
