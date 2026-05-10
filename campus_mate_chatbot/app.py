# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import re
# import json

# app = Flask(__name__)
# CORS(app)

# # -------------------------------
# # 📦 DATA LAYER (Load Responses)
# # -------------------------------
# with open("data.json") as f:
#     RESPONSES = json.load(f)

# # -------------------------------
# # 🧠 NLP PROCESSING LAYER
# # -------------------------------
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
#     tokens = text.split()
#     return tokens

# # -------------------------------
# # 🎯 INTENT RECOGNITION LAYER
# # -------------------------------
# INTENT_PATTERNS = {
#     "greeting": ["hello", "hi", "hey"],
#     "lab_location": ["lab", "location", "where"],
#     "class_timing": ["class", "time", "schedule"],
#     "exam_schedule": ["exam", "date", "test"],
#     "faculty_info": ["faculty", "teacher", "professor", "staff"]
# }

# def detect_intent(tokens):
#     for intent, keywords in INTENT_PATTERNS.items():
#         if any(word in tokens for word in keywords):
#             return intent
#     return "unknown"

# # -------------------------------
# # 💬 RESPONSE GENERATION LAYER
# # -------------------------------
# def generate_response(intent):
#     return RESPONSES.get(intent, RESPONSES["unknown"])

# # -------------------------------
# # 🔗 API LAYER
# # -------------------------------
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_input = data.get("message", "")

#     tokens = preprocess_text(user_input)
#     intent = detect_intent(tokens)
#     response = generate_response(intent)

#     return jsonify({
#         "intent": intent,
#         "response": response
#     })

# # -------------------------------
# # 🚀 RUN SERVER
# # -------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import json
import random

app = Flask(__name__)
CORS(app)
# -------------------------------
# CONTEXT MEMORY
# -------------------------------
last_intent = None

# Load data
with open("data.json") as f:
    RESPONSES = json.load(f)

# NLP
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

# Intent patterns
INTENT_PATTERNS = {
    "greeting": ["hello", "hi", "hey"],
    "lab_location": ["lab", "location", "where"],
    
    # 🔥 NEW
    "lab_timing": ["lab", "time", "timing"],
    
    "class_timing": ["class", "time", "timing", "schedule"],
    "exam_schedule": ["exam", "date", "test"],
    "faculty_info": ["faculty", "teacher", "professor"]
}

# Intent detection
def detect_intent(tokens):
    global last_intent

    # First try normal matching
    for intent, keywords in INTENT_PATTERNS.items():
        if any(word in tokens for word in keywords):
            last_intent = intent
            return intent

    # 🔥 CONTEXT LOGIC (VERY IMPORTANT)
    if "time" in tokens or "timing" in tokens:
        if last_intent == "lab_location":
            return "lab_timing"
        elif last_intent == "class_timing":
            return "class_timing"

    return "unknown"

# Response generation (SAFE)
def generate_response(intent):
    response = RESPONSES.get(intent)

    if not response:
        return RESPONSES["unknown"]

    # support multiple responses
    if isinstance(response, list):
        return random.choice(response)

    return response

# API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "")

        tokens = preprocess_text(user_input)
        intent = detect_intent(tokens)
        response = generate_response(intent)

        return jsonify({
            "response": response
        })

    except Exception as e:
        print("ERROR:", e)   # 🔥 VERY IMPORTANT
        return jsonify({
            "response": "⚠️ Backend error occurred"
        })

if __name__ == "__main__":
    app.run(debug=True)