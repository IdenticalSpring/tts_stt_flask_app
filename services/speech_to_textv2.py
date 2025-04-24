"""
speech_to_text.py
Chuyển file WAV thành văn bản bằng Wav2Vec2-large-960h
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch, soundfile as sf, numpy as np, os, tempfile, uuid

_MODEL = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(_MODEL)
model      = Wav2Vec2ForCTC. from_pretrained(_MODEL)

def transcribe_file(wav_path: str) -> str:
    speech, sr = sf.read(wav_path)
    if speech.ndim > 1:                       # stereo → mono
        speech = speech.mean(axis=1)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    ids   = torch.argmax(logits, dim=-1)
    text  = processor.batch_decode(ids)[0]
    return text.strip()

# CLI test
if __name__ == "__main__":
    demo = "demo.wav"          # đổi tên file test
    print("📢", transcribe_file(demo) if os.path.exists(demo) else "⚠️ Không thấy demo.wav")
