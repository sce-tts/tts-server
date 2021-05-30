import os
import io
from TTS.utils.synthesizer import Synthesizer

path_root = "/content/models/"
synthesizer = Synthesizer(
    path_root + os.environ['TTS_MODEL_FILE'],
    path_root + os.environ['TTS_MODEL_CONFIG'],
    None,
    path_root + os.environ['VOCODER_MODEL_FILE'],
    path_root + os.environ['VOCODER_MODEL_CONFIG'],
    None,
    None,
    False,
)

def synthesize(text):
    wavs = synthesizer.tts(text, None, None)
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    return out