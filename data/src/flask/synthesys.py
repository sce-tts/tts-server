import os
import io
import onnxruntime
import torch
import numpy as np
import time
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.utils.synthesis import synthesis, trim_silence

onnx_vocoder_path = 'VOCODER_MODEL_ONNX' in os.environ and os.environ['VOCODER_MODEL_ONNX']
if onnx_vocoder_path:
    synthesizer = Synthesizer(
        os.environ['TTS_MODEL_FILE'],
        os.environ['TTS_MODEL_CONFIG'],
        None,
        None,
        None,
        None,
        None,
        False,
    )
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    vocoder = onnxruntime.InferenceSession(onnx_vocoder_path, sess_options)
else:
    synthesizer = Synthesizer(
        os.environ['TTS_MODEL_FILE'],
        os.environ['TTS_MODEL_CONFIG'],
        None,
        os.environ['VOCODER_MODEL_FILE'],
        os.environ['VOCODER_MODEL_CONFIG'],
        None,
        None,
        False,
    )

def synthesize(text):
    if onnx_vocoder_path:
        start_time = time.time()
        wavs = []
        sens = synthesizer.split_into_sentences(text)
        for sen in sens:
            _, _, _, mel_postnet_spec, _, _ = synthesis(
                model=synthesizer.tts_model,
                text=sen,
                CONFIG=synthesizer.tts_config,
                use_cuda=synthesizer.use_cuda,
                ap=synthesizer.ap,
                speaker_id=None,
                style_wav=None,
                truncated=False,
                enable_eos_bos_chars=synthesizer.tts_config.enable_eos_bos_chars,
                use_griffin_lim=False,
                speaker_embedding=None,
            )
            mel_postnet_spec = synthesizer.ap.denormalize(mel_postnet_spec.T).T
            vocoder_input = synthesizer.ap.normalize(mel_postnet_spec.T)
            vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)

            inference_padding = 5
            padded_vocoder_input = torch.nn.functional.pad(vocoder_input, (inference_padding, inference_padding), "replicate")
            ort_inputs = {vocoder.get_inputs()[0].name: padded_vocoder_input.detach().numpy()}
            ort_outs = vocoder.run(None, ort_inputs)
            waveform = np.squeeze(ort_outs[0])
            waveform = trim_silence(waveform, synthesizer.ap)
            wavs += list(waveform)
            wavs += [0] * 10000
        process_time = time.time() - start_time
        audio_time = len(wavs) / synthesizer.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
    else:
        wavs = synthesizer.tts(text, None, None)
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    return out