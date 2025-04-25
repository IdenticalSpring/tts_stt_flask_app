from flask import Flask, request, render_template, send_file, redirect, url_for
from services.speech_to_textv2 import start_recording, stop_and_transcribe
from services.text_to_speechv2 import synth_bytes
import io

app = Flask(__name__)

text_result = ""
last_audio = None  # lưu WAV vào RAM để phát lại

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", text_result=text_result, audio_ready=bool(last_audio))

@app.route("/start-record", methods=["POST"])
def start():
    start_recording()
    return redirect(url_for('index'))

@app.route("/stop-record", methods=["POST"])
def stop():
    global text_result
    text_result = stop_and_transcribe()
    return redirect(url_for('index'))

@app.route("/tts", methods=["POST"])
def tts():
    global last_audio
    text = request.form["text"]
    last_audio = synth_bytes(text)
    return redirect(url_for('index'))

@app.route("/audio")
def audio():
    global last_audio
    if last_audio:
        return send_file(
            io.BytesIO(last_audio),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="output.wav"
        )
    return "No audio generated yet", 404

if __name__ == "__main__":
    app.run(debug=True)
