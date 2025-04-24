from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch


def text_to_speech(input):
    pipeline = KPipeline(lang_code='a')
    text = {input}
    generator = pipeline(text, voice='af_heart')

    # luu WAV
    for i, (gs, ps, audio) in enumerate(generator):
        print(i, gs, ps)
        display(Audio(data=audio, rate=24000, autoplay=i==0))
        sf.write(f'{i}.wav', audio, 24000)

    return audio