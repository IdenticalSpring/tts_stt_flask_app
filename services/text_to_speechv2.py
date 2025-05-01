"""
tts_play_cli.py
Nhập text → sinh giọng Kokoro → phát bằng winsound → xoá file tạm.
Chạy được trên Windows mà không cần cài trình biên dịch C++.
"""
import io, os, tempfile, numpy as np, soundfile as sf, torch, winsound
from kokoro import KPipeline

pipe  = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
DEFAULT_VOICE = "af_bella"  # mặc định nếu không truyền voice


def list_voices() -> list[str]:
    return pipe.available_voices()


def synth_bytes(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """ Sinh WAV bytes (PCM 24 kHz) – tương thích mọi phiên bản Kokoro """
    _, _, audio = next(pipe(text, voice=voice))
    
    if isinstance(audio, dict):               # API cũ
        arr = np.asarray(audio["array"], dtype="float32")
        sr  = int(audio["sampling_rate"])
    elif isinstance(audio, torch.Tensor):     # API mới
        arr = audio.cpu().numpy().astype("float32")
        sr  = 24_000
    else:
        raise TypeError("audio phải là dict hoặc Tensor")

    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    buf = io.BytesIO(); sf.write(buf, arr, sr, format="WAV")
    return buf.getvalue()


def play_once(text: str, voice: str = DEFAULT_VOICE):
    wav_bytes = synth_bytes(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name
    winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    os.remove(wav_path)


if __name__ == "__main__":
    try:
        while True:
            line = input("Nhập câu (Enter trống để thoát): ").strip()
            if not line:
                break
            play_once(line)
    except KeyboardInterrupt:
        pass
