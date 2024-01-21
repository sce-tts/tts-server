import os
import io
import wave
from piper import PiperVoice

voice = PiperVoice.load(
    os.environ["TTS_MODEL_FILE"],
    config_path=os.environ["TTS_MODEL_CONFIG"],
    use_cuda=False,
)


def synthesize(text: str, wav_io: io.BytesIO) -> bytes:
    with wave.open(wav_io, "wb") as wav_file:
        voice.synthesize(text=text, wav_file=wav_file)
    return wav_io.getvalue()
