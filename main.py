from flask import Flask, request, jsonify, send_file
import tempfile, os
from speech_to_text import transcribe_file
from text_to_speech import synthesize

app = Flask(__name__)

# -------- Speech-to-Text --------
@app.route("/speech-to-text", methods=["POST"])
def stt_route():
    if "file" not in request.files:
        return jsonify(error="Thiếu file âm thanh (.wav)"), 400
    f = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        f.save(tmp.name)
    try:
        text = transcribe_file(tmp.name)
        return jsonify(text=text)
    finally:
        os.remove(tmp.name)

# -------- Text-to-Speech --------
@app.route("/text-to-speech", methods=["POST"])
def tts_route():
    data = request.get_json(silent=True) or {}
    if "text" not in data:
        return jsonify(error="Thiếu trường 'text'"), 400
    wav_path = synthesize(data["text"])
    return send_file(wav_path, as_attachment=True, download_name="tts.wav", mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
