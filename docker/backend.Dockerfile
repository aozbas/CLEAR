FROM python:3.13-slim

WORKDIR /workspace

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/backend-requirements.txt /tmp/backend-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
    && python -m pip install -r /tmp/backend-requirements.txt

COPY backend ./backend
COPY ml ./ml

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
