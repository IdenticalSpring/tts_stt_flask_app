from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
from services.task_manager import add_task
from services.text_to_speechv2 import synth_bytes, list_voices
import base64
import uuid
import os
import io

app = Flask(__name__)

# === Trang chủ ===
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# === TTS: Nhận văn bản, trả về .wav (stream) ===
@app.route("/tts", methods=["POST"])
def tts():
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get("text")
            voice = data.get("voice", "af_bella")
        else:
            text = request.form.get("text")
            voice = request.form.get("voice", "af_bella")

        if not text:
            return jsonify({"status": "error", "message": "Missing text input"}), 400

        wav_bytes = synth_bytes(text, voice)
        base64_audio = base64.b64encode(wav_bytes).decode("utf-8")
        return jsonify({
            "status": "success",
            "audioData": base64_audio,
            "voiceID": voice
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  
        return jsonify({"status": "error", "message": str(e)}), 500



# === Trả danh sách voice ===
@app.route("/voices")
def voices():
    return jsonify(voices=list_voices())


# === Khởi tạo thư mục tạm và chạy ===
if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True)
