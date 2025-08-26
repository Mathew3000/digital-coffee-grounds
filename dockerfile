# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# SciPy works from wheels on slim; add a couple libs used by matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
      fonts-dejavu-core \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY static/favicon.ico /app/static/favicon.ico

EXPOSE 5000
CMD ["python", "app.py"]
