"""
Flask API for Speech Emotion Recognition + serves the built React app.

Dev:   python app.py            (API on http://127.0.0.1:5000)
       cd frontend && npm run dev   (React UI on http://localhost:5173, proxies /api)
Prod:  cd frontend && npm run build, then python app.py  -> everything on http://127.0.0.1:5000
"""
import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
DIST = os.path.join(BASE, "frontend", "dist")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
app.json.sort_keys = False  # keep probabilities ordered highest first

_predictor = None
_lock = threading.Lock()


def get_predictor():
    global _predictor
    with _lock:
        if _predictor is None:
            from predict import EmotionPredictor
            _predictor = EmotionPredictor(MODEL_DIR)
        return _predictor


@app.after_request
def allow_cors(resp):
    # lets the React dev server call the API directly too (not only through the Vite proxy)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/api/status")
def status():
    try:
        p = get_predictor()
        return jsonify(ready=True, labels=p.labels, metrics=p.metrics)
    except FileNotFoundError as e:
        return jsonify(ready=False, reason="untrained", message=str(e))
    except Exception as e:  # noqa: BLE001  (e.g. a package from requirements.txt is missing)
        return jsonify(ready=False, reason="error", message=f"{type(e).__name__}: {e}")


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return ("", 204)
    f = request.files.get("audio")
    if f is None or f.filename == "":
        return jsonify(error="No audio received. Record a clip or choose a file."), 400
    try:
        p = get_predictor()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"Model could not load ({type(e).__name__}: {e})."), 503

    suffix = os.path.splitext(f.filename)[1] or ".wav"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        f.save(tmp)
        with _lock:
            result = p.predict(tmp)
        return jsonify(result)
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"Could not read this audio ({e}). Try a WAV file."), 400
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File is larger than 25 MB. Use a shorter clip."), 413


# ---------- React build ----------
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def spa(path):
    if not os.path.isfile(os.path.join(DIST, "index.html")):
        return (
            "<h2>React UI not built yet</h2>"
            "<p>Either run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run dev</code> "
            "and open <a href='http://localhost:5173'>http://localhost:5173</a>, "
            "or run <code>npm run build</code> and reload this page.</p>",
            200,
        )
    if path and os.path.isfile(os.path.join(DIST, path)):
        return send_from_directory(DIST, path)
    return send_from_directory(DIST, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"API running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
