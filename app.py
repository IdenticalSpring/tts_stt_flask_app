from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
from services.speech_to_textv2 import start_recording, stop_and_transcribe
from services.task_manager import add_task
from services.text_to_speechv2 import synth_bytes, list_voices

import uuid
import os
import io

app = Flask(__name__)

# === Trang chủ ===
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# === Bắt đầu ghi âm ===
@app.route("/start-record", methods=["POST"])
def start():
    start_recording()
    return redirect(url_for('index'))


# === Hàm phụ để lưu text từ STT ===
def transcribe_and_store_to_file(task_id):
    text_result = stop_and_transcribe()
    with open(f"temp/text_{task_id}.txt", "w", encoding="utf-8") as f:
        f.write(text_result)


# === Dừng ghi âm và xử lý text ===
@app.route("/stop-record", methods=["POST"])
def stop():
    task_id = str(uuid.uuid4())
    add_task(transcribe_and_store_to_file, task_id)
    return redirect(url_for('result', task_id=task_id))


# === Trả kết quả STT ===
@app.route("/result/<task_id>")
def result(task_id):
    text_path = f"temp/text_{task_id}.txt"

    if not os.path.exists(text_path):
        return render_template("processing.html", task_id=task_id)

    with open(text_path, encoding="utf-8") as f:
        text_result = f.read()

    return render_template("result.html", text_result=text_result)


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

        return send_file(
            io.BytesIO(wav_bytes),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="tts_output.wav"
        )

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
