docker-compose run --rm flask python /content/src/TTS/TTS/bin/convert_hifigan_onnx.py ^
    --torch_model_path "/content/models/hifigan-v2/best_model_291688.pth.tar" ^
    --config_path "/content/models/hifigan-v2/config.json" ^
    --output_path "/content/models/hifigan-v2/hifigan.onnx"
pause