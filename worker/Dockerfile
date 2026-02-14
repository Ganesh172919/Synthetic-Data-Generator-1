FROM python:3.11-slim

WORKDIR /app

COPY worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY Pre-Work ./Pre-Work
COPY worker ./worker

ENV DATA_DIR=/data
ENV SQLITE_PATH=/data/synthgen.sqlite
ENV OUTPUTS_DIR=/data/outputs
ENV POLL_INTERVAL_MS=1000
ENV PROGRESS_UPDATE_INTERVAL_MS=2000
ENV MAX_CONCURRENT_JOBS=1

CMD ["python", "worker/main.py"]
