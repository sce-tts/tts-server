FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    unzip \
    git \
    espeak-ng \
    libsndfile1-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "konlpy" \ 
    "jamo" \ 
    "nltk" \
    "python-mecab-ko" \
    "onnxruntime" \
    "flask"

RUN mkdir -p /content/src

WORKDIR /content/src

RUN git clone --depth 1 https://github.com/sce-tts/g2pK.git
RUN git clone --depth 1 https://github.com/sce-tts/TTS.git -b sce-tts

WORKDIR /content/src/g2pK
RUN pip install --no-cache-dir -e .

WORKDIR /content/src/TTS
RUN pip install --no-cache-dir -e .

RUN mkdir -p /content/src/flask
WORKDIR /content/src/flask

EXPOSE 5000

CMD ["python", "-u", "server.py"]