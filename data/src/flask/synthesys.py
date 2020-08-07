import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # USE CPU

import yaml
import json
import numpy as np
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

sys.path.append('/content/src/TensorflowTTS')
from tensorflow_tts.processor.ljspeech import LJSpeechProcessor
from tensorflow_tts.processor.ljspeech import symbols as tensorflowtts_symbols
from tensorflow_tts.processor.ljspeech import _symbol_to_id

# from tensorflow_tts.configs import FastSpeech2Config
# from tensorflow_tts.models import TFFastSpeech2

from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig
from tensorflow_tts.models import TFMelGANGenerator
from tensorflow_tts.models import TFPQMF
sys.path.remove('/content/src/TensorflowTTS')

sys.path.append('/content/src/glow-tts')
from utils import HParams, load_checkpoint
from text import symbols, text_to_sequence
from audio_processing import dynamic_range_decompression
from models import FlowGenerator
sys.path.remove('/content/src/glow-tts')

SAMPLING_RATE = 22050

def load_glow_tts(config_path, checkpoint_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    model = FlowGenerator(
        len(symbols),
        out_channels=hparams.data.n_mel_channels,
        **hparams.model
    ).to("cpu")

    load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()

    return model

def inference_glow_tts(text, model, noise_scale=0.333, length_scale=0.9):
    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).cpu()
    with torch.no_grad():
        (y_gen_tst, *r), attn_gen, *_ = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
    return y_gen_tst

def convert_mel(mel):
    converted = mel.float().data.cpu().numpy()
    converted = np.expand_dims(np.transpose(converted[0]), axis=0)
    return converted

def normalize_mel(mel, mean, sigma):
    normalized = dynamic_range_decompression(mel)
    normalized = convert_mel(normalized)
    normalized = (np.log10(normalized) - mean) / sigma
    return normalized

def load_fastspeech2(config_path, model_path):
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        fs2_config = FastSpeech2Config(**raw_config["fastspeech_params"])
        fastspeech2 = TFFastSpeech2(config=fs2_config, name="fastspeech")
        fastspeech2._build()
        fastspeech2.load_weights(model_path, by_name=True)
    return fastspeech2

def inference_fastspeech2(text, model):
    input_ids = processor.text_to_sequence(text)
    input_ids = np.concatenate([input_ids, [len(tensorflowtts_symbols) - 1]], -1)
    
    mel_before, mel_after, duration_outputs, _, _ = model.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        attention_mask=tf.math.not_equal(tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32)
    )
    return mel_after

def load_mb_melgan(config_path, model_path):
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        mb_melgan_config = MultiBandMelGANGeneratorConfig(**raw_config["generator_params"])
        mb_melgan = TFMelGANGenerator(config=mb_melgan_config, name='melgan_generator')
        mb_melgan._build()
        mb_melgan.load_weights(model_path)
        pqmf = TFPQMF(config=mb_melgan_config, name="pqmf")
    return (mb_melgan, pqmf)

def load_stats(stats_path):
    mean, scale = np.load(stats_path)
    sigma = np.sqrt(scale)
    return mean, sigma

def synthesis(mb_melgan, pqmf, mel):
    generated_subbands = mb_melgan(mel)
    generated_audios = pqmf.synthesis(generated_subbands)
    return generated_audios[0, :, 0]


glow_tts = load_glow_tts(
    config_path='/content/models/glow-tts/config.json',
    checkpoint_path=f'/content/models/glow-tts/G_{os.environ.get("TTS_GLOW_TTS")}.pth'
)

processor = LJSpeechProcessor(None, "korean_cleaners")

# fastspeech2 = load_fastspeech2(
#     config_path='/content/models/fastspeech2/config.yml',
#     model_path=f'/content/models/fastspeech2/checkpoints/model-{os.environ.get("TTS_FASTSPEECH2")}.h5'
# )

mb_mean, mb_sigma = load_stats(
    stats_path='/content/models/mb-melgan/stats.npy'
)

mb_melgan, pqmf = load_mb_melgan(
    config_path='/content/models/mb-melgan/config.yml',
    model_path=f'/content/models/mb-melgan/checkpoints/generator-{os.environ.get("TTS_MULTIBAND_MELGAN")}.h5'
)

def generate_audio_glow_tts(text):
    mel_original = inference_glow_tts(text, glow_tts)
    mel_nomalized = normalize_mel(mel_original, mb_mean, mb_sigma)
    audio = synthesis(mb_melgan, pqmf, mel_nomalized)
    return audio

def generate_audio_fastspeech2(text):
    mel = inference_fastspeech2(text, fastspeech2)
    audio = synthesis(mb_melgan, pqmf, mel)
    return audio