import io
import re
import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

# Khởi tạo pipeline Kokoro
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

# Load tất cả voice sẵn có
voice_list = [
    "af_heart", "af_alloy", "af_aoede", "af_jessica", "af_kore", "af_nicole", 
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", 
    "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"
]

for v in voice_list:
    try:
        pipe.load_voice(v)
    except Exception as e:
        print(f"Không load được voice: {v} – {e}")

DEFAULT_VOICE = "af_bella"

def list_voices() -> list[str]:
    return list(pipe.voices.keys())

def split_text(text: str) -> list[str]:
    # Cắt văn bản dài thành các câu nhỏ dựa vào dấu câu
    return [s.strip() for s in re.split(r'[.!?;]', text) if s.strip()]

def synth_with_pause(text: str, voice: str = DEFAULT_VOICE, pause_ms: int = 400) -> bytes:
    sentences = split_text(text)
    audios = []
    silence = np.zeros(int(24000 * (pause_ms / 1000.0)), dtype=np.float32)

    for sent in sentences:
        try:
            _, _, audio = next(pipe(sent, voice=voice))
        except Exception as e:
            print(f"Lỗi synth câu: {sent} – {e}")
            continue

        if isinstance(audio, dict):
            arr = np.asarray(audio["array"], dtype="float32")
        elif isinstance(audio, torch.Tensor):
            arr = audio.cpu().numpy().astype("float32")
        else:
            raise TypeError("audio phải là dict hoặc Tensor")

        audios.append(arr)
        audios.append(silence)

    if not audios:
        raise RuntimeError("Không synth được bất kỳ câu nào.")

    full_audio = np.concatenate(audios)
    buf = io.BytesIO()
    sf.write(buf, full_audio.reshape(-1, 1), 24000, format="WAV")
    return buf.getvalue()

# === ALIAS để Flask import bình thường ===
synth_bytes = synth_with_pause
