FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md openenv.yaml ./
COPY __init__.py client.py environment.py models.py tasks.py ./
COPY server ./server

RUN python -m pip install --upgrade pip && \
    pip install .

EXPOSE 8000

CMD ["python", "-m", "server.app"]
