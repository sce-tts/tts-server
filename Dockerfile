FROM python:3.10.13-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -U pip setuptools wheel poetry

WORKDIR /data
COPY pyproject.toml poetry.lock /data/
RUN pip3 install --no-cache-dir poetry \
    && poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && pip3 install --no-cache-dir -r requirements.txt
COPY data/src /data/src

EXPOSE 5000

CMD ["python", "-u", "/data/src/main.py"]