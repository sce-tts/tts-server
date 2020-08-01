FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
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
    "h5py>=2.10.0" \
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
    "flask"

RUN curl https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash

RUN mkdir -p /content/src/flask
WORKDIR /content/src/flask

EXPOSE 8888

CMD ["python", "server.py"]