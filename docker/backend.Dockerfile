FROM python:3.11-slim

WORKDIR /plant_detect2

RUN pip install --upgrade pip setuptools wheel
COPY backend/requirements.txt requirements.txt
COPY models models
RUN pip install -r requirements.txt


COPY backend/ backend/

ENV PYTHONPATH=/plant_detect2/backend/app/src


# MODEL_PATH: where downloaded model files are cached inside the container
ENV MODEL_PATH=/models/wandb
ENV WANDB_API_KEY="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"
ENV WANDB_PROJECT="certification"

CMD uvicorn backend.app.api.main:api --host 0.0.0.0 --port ${PORT:-8080}