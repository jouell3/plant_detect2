FROM python:3.11-slim

WORKDIR /plant_detect2

RUN pip install --upgrade pip setuptools wheel
COPY backend/requirements.txt requirements.txt
COPY models models
RUN pip install -r requirements.txt


COPY backend/ backend/

ENV PYTHONPATH=/plant_detect2/backend/app/src


# MODEL_PATH: where downloaded model files are cached inside the container
ENV MODEL_PATH=/plant_detect2/models/wandb

CMD uvicorn backend.app.api.main:api --host 0.0.0.0 --port ${PORT:-8080}