import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # USE CPU

import yaml
import json
import numpy as np
import torch

sys.path.append("/content/src/TensorflowTTS")
from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig
from tensorflow_tts.models import TFMelGANGenerator
from tensorflow_tts.models import TFPQMF

sys.path.remove("/content/src/TensorflowTTS")

sys.path.append("/content/src/glow-tts")
from utils import HParams, load_checkpoint
from text import symbols, text_to_sequence
from audio_processing import dynamic_range_decompression
from models import FlowGenerator

sys.path.remove("/content/src/glow-tts")

SAMPLING_RATE = 22050


def load_glow_tts(config_path, checkpoint_path):
    with torch.no_grad():
        with open(config_path, "r") as f:
            data = f.read()
        config = json.loads(data)

        hparams = HParams(**config)
        model = FlowGenerator(
            len(symbols), out_channels=hparams.data.n_mel_channels, **hparams.model
        ).to("cpu")

        load_checkpoint(checkpoint_path, model)
        model.decoder.store_inverse()  # do not calcuate jacobians for fast decoding
        model.eval()

    return model


def inference_glow_tts(text, model, noise_scale=0.333, length_scale=0.9):
    with torch.no_grad():
        sequence = np.array(text_to_sequence(text, ["korean_cleaners"]))[None, :]
        x = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
        x_lengths = torch.tensor([x.shape[1]]).cpu()
        (y_gen, *r), attn_gen, *_ = model(
            x, x_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale,
        )
    return y_gen


def convert_mel(mel):
    with torch.no_grad():
        converted = mel.float().data.cpu().numpy()
        converted = np.expand_dims(np.transpose(converted[0]), axis=0)
    return converted


def normalize_mel(mel, mean, sigma):
    normalized = dynamic_range_decompression(mel)
    normalized = convert_mel(normalized)
    normalized = (np.log10(normalized) - mean) / sigma
    return normalized


def load_mb_melgan(config_path, model_path):
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        mb_melgan_config = MultiBandMelGANGeneratorConfig(
            **raw_config["generator_params"]
        )
        mb_melgan = TFMelGANGenerator(config=mb_melgan_config, name="melgan_generator")
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
    config_path="/content/models/glow-tts/config.json",
    checkpoint_path=f'/content/models/glow-tts/G_{os.environ.get("TTS_GLOW_TTS")}.pth',
)

mb_mean, mb_sigma = load_stats(stats_path="/content/models/mb-melgan/stats.npy")

mb_melgan, pqmf = load_mb_melgan(
    config_path="/content/models/mb-melgan/config.yml",
    model_path=f'/content/models/mb-melgan/checkpoints/generator-{os.environ.get("TTS_MULTIBAND_MELGAN")}.h5',
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
