from flask import Flask, jsonify, request
from flask_cors import CORS

from config import Config
from rag_service import answer_question

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": Config.CLIENT_ORIGIN}})


@app.get("/api/health")
def health():
    """Confirm that the backend is running."""
    return jsonify(
        {
            "status": "ok",
            "message": "Backend is running.",
        }
    ), 200


@app.post("/api/ask")
def ask_question():
    """Receive a question from the frontend and return an answer with sources."""
    # Read JSON from the request body.
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    # Validate that the question exists and is not blank.
    # Return a helpful error response if the question is missing.
    if not question:
        return jsonify({"error": "Question is required."}), 400

    # Call answer_question(question).
    result = answer_question(question)

    # Return the result as JSON.
    return jsonify(result), 200



if __name__ == "__main__":
    app.run(debug=True, port=5555)
