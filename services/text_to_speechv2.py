"""
text_to_speech.py
Sinh giọng nói từ văn bản với Kokoro (hexgrad/Kokoro-82M)
"""

from kokoro import KPipeline
import numpy as np, soundfile as sf, os, uuid

_REPO  = "hexgrad/Kokoro-82M"   # gọi thẳng repo gốc để khỏi lằng nhằng
_VOICE = "af_bella"             # đổi sang voice tồn tại trong repo

pipe = KPipeline(lang_code="a", repo_id=_REPO)

def synthesize(text: str, out_dir="tts_outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    _, _, audio = next(pipe(text, voice=_VOICE))   # lấy 1 kết quả
    arr  = np.asarray(audio["array"]).astype("float32")
    if arr.ndim == 1:                               # bảo đảm (N,1)
        arr = arr.reshape(-1, 1)
    path = os.path.join(out_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(path, arr, audio["sampling_rate"])
    return path

# CLI test
if __name__ == "__main__":
    print("✅ File tạo:", synthesize("i am gay"))
