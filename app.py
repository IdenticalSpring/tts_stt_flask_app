from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
from services.speech_to_textv2 import start_recording, stop_and_transcribe
from services.task_manager import add_task
from services.text_to_speechv2 import synth_bytes, list_voices  # ✅ sửa chỗ này
import uuid
import os
import io
import base64
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/start-record", methods=["POST"])
def start():
    start_recording()
    return redirect(url_for('index'))

def transcribe_and_store_to_file(task_id):
    text_result = stop_and_transcribe()  # 🔥 CHỈ 1 giá trị trả về
    with open(f"temp/text_{task_id}.txt", "w", encoding="utf-8") as f:
        f.write(text_result)

@app.route("/stop-record", methods=["POST"])
def stop():
    task_id = str(uuid.uuid4())
    add_task(transcribe_and_store_to_file, task_id)
    return redirect(url_for('result', task_id=task_id))

@app.route("/result/<task_id>")
def result(task_id):
    text_path = f"temp/text_{task_id}.txt"

    if not os.path.exists(text_path):
        return render_template("processing.html", task_id=task_id)

    with open(text_path, encoding="utf-8") as f:
        text_result = f.read()

    return render_template("result.html", text_result=text_result)


last_audio = None  # Global lưu WAV tạm thời khi TTS



@app.route("/tts", methods=["POST"])
def tts():
    text = request.form.get("text") or request.json.get("text")
    voice = request.form.get("voice") or request.json.get("voice", "af_bella")

    if not text:
        return jsonify({"status": "error", "message": "Missing text input"}), 400

    try:
        wav_bytes = synth_bytes(text, voice)
        base64_audio = base64.b64encode(wav_bytes).decode("utf-8")

        return jsonify({
            "status": "success",
            "audioData": base64_audio,
            "voiceID": voice
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/tts-result")
def tts_result():
    global last_audio
    if last_audio:
        return send_file(
            io.BytesIO(last_audio),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="tts_output.wav"
        )
    else:
        return "No TTS audio generated yet", 404

@app.route("/voices")
def voices():
    return jsonify(voices=list_voices())  # ✅ trả list voice dưới dạng JSON

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True)
