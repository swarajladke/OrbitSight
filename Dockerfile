FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml .
COPY run.sh .

RUN chmod +x run.sh

ENV PYTHONUNBUFFERED=1

CMD ["sh", "run.sh"]
