FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    espeak \
    espeak-ng \
    g++ \
    gcc \
    git \
    libsndfile1-dev \
    make \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install -U pip setuptools wheel

WORKDIR /data
COPY pyproject.toml poetry.lock /data/
RUN pip install --no-cache-dir poetry python-mecab-ko \
    && poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && pip install --no-cache-dir -r requirements.txt
COPY data/src /data/src

EXPOSE 5000

CMD ["python", "-u", "/data/src/main.py"]