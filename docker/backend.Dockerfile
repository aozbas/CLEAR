FROM python:3.13-slim

WORKDIR /workspace

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements*.txt /tmp/
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
    && grep -v -E '^(torch|torchvision)==' /tmp/requirements.txt > /tmp/backend-runtime-requirements.txt \
    && python -m pip install -r /tmp/backend-runtime-requirements.txt

RUN groupadd --system clear \
    && useradd --system --gid clear --home-dir /nonexistent --shell /usr/sbin/nologin clear

COPY --chown=clear:clear backend ./backend
COPY --chown=clear:clear ml/__init__.py ./ml/__init__.py
COPY --chown=clear:clear ml/preprocessing.py ./ml/preprocessing.py
COPY --chown=clear:clear ml/inference ./ml/inference
COPY --chown=clear:clear ml/models ./ml/models

USER clear

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-server-header"]
