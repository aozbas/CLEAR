FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY docker/ml-requirements.txt /tmp/ml-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/ml-requirements.txt

COPY ml ./ml

CMD ["python", "-m", "ml.training.train", "--help"]
