import threading
import numpy as np
import sounddevice as sd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

SAMPLE_RATE = 16_000
MODEL_NAME  = "facebook/wav2vec2-large-960h"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2ForCTC   .from_pretrained(MODEL_NAME)

_recording = False
_chunks = []
_stream = None

def start_recording():
    global _recording, _chunks, _stream
    _chunks = []
    _recording = True

    def callback(indata, frames, time, status):
        if _recording:
            _chunks.append(indata.copy())

    _stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32", callback=callback)
    _stream.start()

def stop_and_transcribe() -> str:
    global _recording, _stream
    _recording = False
    _stream.stop()
    _stream.close()

    audio = np.concatenate(_chunks, axis=0).flatten()
    print(f"🎧 Đã thu {len(audio) / SAMPLE_RATE:.1f}s, bắt đầu nhận dạng...")

    inputs = processor(audio, sampling_rate=SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        pred = model(**inputs).logits.argmax(dim=-1)

    text = processor.decode(pred[0])
    return text.strip()

# --- CLI test ---
if __name__ == "__main__":
    start_recording()
    input("🎤 Đang ghi âm… nhấn ENTER để dừng\n")
    result = stop_and_transcribe()
    print("📄 Kết quả STT:", result)
