from flask import Flask, render_template, request, jsonify, Response
import os
import uuid

from tools.llm.llm_client import get_llm_client
from tools.data_ingestion.parser import handle_file_upload
from tools.context_management.context_handler import read_context, write_context
from tools.llm.orchestrator import process_user_message, cotas_generate_insights

_llm_client = get_llm_client()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

BASE_STORAGE = "storage"


def get_session_path(session_id: str) -> dict:
    session_path = os.path.join(BASE_STORAGE, session_id)
    return {
        "session": session_path,
        "datasets": os.path.join(session_path, "datasets"),
        "scripts": os.path.join(session_path, "scripts"),
        "results": os.path.join(session_path, "results"),
        "context": os.path.join(session_path, "context.json"),
        "dataset_metadata": os.path.join(session_path, "dataset_metadata.json"),
        "chat_history": os.path.join(session_path, "chat_history.json"),
        "cotas_log": os.path.join(session_path, "cotas_log.json"),
    }


def ensure_session_dirs(paths: dict):
    os.makedirs(paths["session"], exist_ok=True)
    os.makedirs(paths["datasets"], exist_ok=True)
    os.makedirs(paths["scripts"], exist_ok=True)
    os.makedirs(paths["results"], exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/create-session", methods=["POST"])
def create_session():
    session_id = str(uuid.uuid4())
    paths = get_session_path(session_id)
    ensure_session_dirs(paths)

    write_context(paths["context"], {"session_id": session_id, "history": []})
    write_context(paths["dataset_metadata"], {"datasets": []})
    write_context(paths["chat_history"], {"messages": []})

    return jsonify({"session_id": session_id, "message": "Session created successfully"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id")
    user_message = data.get("message", "").strip()

    if not session_id or not user_message:
        return jsonify({"error": "session_id and message required"}), 400

    paths = get_session_path(session_id)
    if not os.path.exists(paths["session"]):
        return jsonify({"error": "Invalid session_id"}), 404

    try:
        response = process_user_message(user_message, paths)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/upload-file", methods=["POST"])
def upload_file():
    session_id = request.form.get("session_id")
    file = request.files.get("file")

    if not session_id or not file:
        return jsonify({"error": "session_id and file required"}), 400

    paths = get_session_path(session_id)
    if not os.path.exists(paths["session"]):
        return jsonify({"error": "Invalid session_id"}), 404

    try:
        metadata = handle_file_upload(
            file,
            paths["datasets"],
            paths["dataset_metadata"],
            paths["context"],
        )
        return jsonify({"message": "File uploaded and processed", "metadata": metadata})
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route("/generate-insights", methods=["POST"])
def generate_insights():
    data = request.get_json()
    session_id = data.get("session_id")
    user_goal = data.get("goal", "Perform comprehensive data analysis")
    max_loops = data.get("max_loops", 15)

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    paths = get_session_path(session_id)
    if not os.path.exists(paths["session"]):
        return jsonify({"error": "Invalid session_id"}), 404

    def generate():
        try:
            for update in cotas_generate_insights(paths, user_goal, max_loops):
                yield f"data: {update}\n\n"
        except Exception as e:
            yield f'data: {{"error": "Analysis failed: {str(e)}"}}\n\n'

    return Response(generate(), mimetype="text/event-stream")


@app.route("/get-context", methods=["GET"])
def get_context():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    paths = get_session_path(session_id)
    if not os.path.exists(paths["session"]):
        return jsonify({"error": "Invalid session_id"}), 404

    context = read_context(paths["context"])
    metadata = read_context(paths["dataset_metadata"])
    return jsonify({"context": context, "dataset_metadata": metadata})


@app.route("/get-chat-history", methods=["GET"])
def get_chat_history():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    paths = get_session_path(session_id)
    if not os.path.exists(paths["session"]):
        return jsonify({"error": "Invalid session_id"}), 404

    chat_history = read_context(paths["chat_history"])
    return jsonify(chat_history)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
