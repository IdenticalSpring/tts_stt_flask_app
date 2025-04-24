"""
speech_to_text_record_mem.py
Ghi âm micro (Enter để dừng) ➜ chuyển thẳng sang text (không ghi đĩa)
"""

import threading, numpy as np, sounddevice as sd, torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

SAMPLE_RATE = 16_000
MODEL_NAME  = "facebook/wav2vec2-large-960h"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2ForCTC   .from_pretrained(MODEL_NAME)

# ---------- Ghi âm không giới hạn ----------
chunks, recording = [], True

def callback(indata, frames, time, status):
    chunks.append(indata.copy())

def wait_enter():
    global recording
    input("Đang ghi âm… nhấn ENTER để dừng\n")
    recording = False

threading.Thread(target=wait_enter, daemon=True).start()

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32",
                    callback=callback):
    while recording:
        sd.sleep(100)

audio = np.concatenate(chunks, axis=0).flatten()   # float32 1-D
print(f"Đã thu {len(audio)/SAMPLE_RATE:.1f}s, bắt đầu nhận dạng…")

# ---------- STT trực tiếp từ mảng numpy ----------
inputs = processor(audio, sampling_rate=SAMPLE_RATE,
                   return_tensors="pt", padding=True)

with torch.no_grad():
    pred = model(**inputs).logits.argmax(dim=-1)

text = processor.decode(pred[0])
print("Kết quả STT:", text)
