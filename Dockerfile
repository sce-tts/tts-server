FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
ENV NVIDIA_VISIBLE_DEVICES all
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    unzip \
    git \
    libsndfile1 \
    g++ \
    default-jdk \
    libicu-dev \ 
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "cython==0.29.12" \ 
    "tensorflow-gpu>=2.2.0" \
    "tensorflow-addons>=0.9.1" \
    "setuptools>=38.5.1" \
    "librosa>=0.7.0" \
    "soundfile>=0.10.2" \
    "matplotlib>=3.1.0" \
    "PyYAML>=3.12" \
    "tqdm>=4.26.1" \
    "h5py==2.10.0" \
    "pathos>=0.2.5" \
    "unidecode>=1.1.1" \
    "inflect>=4.1.0" \
    "scikit-learn>=0.22.0" \
    "pyworld>=0.2.10" \
    "numba<=0.48" \
    "numpy" \
    "scipy" \
    "pillow" \
    "future" \ 
    "konlpy" \ 
    "jamo" \ 
    "nltk" \
    "python-mecab-ko" \
    "flask"

RUN mkdir -p /content/src

WORKDIR /content/src

RUN git clone --depth 1 https://github.com/sce-tts/g2pK.git
RUN git clone --depth 1 https://github.com/sce-tts/glow-tts.git
RUN git clone --depth 1 https://github.com/sce-tts/TensorflowTTS.git -b r0.7

WORKDIR /content/src/glow-tts/monotonic_align
RUN python setup.py build_ext --inplace

ADD data/src/load_g2pk.py /content/src/load_g2pk.py
WORKDIR /content/src
RUN python load_g2pk.py

RUN mkdir -p /content/src/flask
WORKDIR /content/src/flask

EXPOSE 5000

CMD ["python", "-u", "server.py"]