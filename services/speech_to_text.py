from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


def speech_to_text(input):
    # Cấu hình
    sample_rate = 16000  # Tần số mẫu mà model yêu cầu

    # Tải model và processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    # Thu âm từ micro
    def record_voice(duration=5):
        print("Bắt đầu ghi âm trong", duration, "giây...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Chờ cho ghi âm hoàn tất
        print("Ghi âm hoàn tất.")
        return np.squeeze(recording)

    # Chuyển đổi giọng nói thành văn bản
    def transcribe_audio(audio):
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        transcription = processor.batch_decode(logits.argmax(dim=-1))[0]
        return transcription

    # Quy trình hoàn chỉnh
    audio_data = record_voice(duration=15)  # Ghi âm 5 giây (bạn có thể thay đổi thời gian)
    text = transcribe_audio(audio_data)
    print("Transcription:", text)
