FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/scorer_pregeom.joblib ./models/
COPY models/scorer_objectness_pre_geometry.joblib ./models/
COPY models/box_regressor_arm2.joblib ./models/
COPY models/model_structure.json ./models/
COPY config.yaml .
COPY run.sh .

RUN chmod +x run.sh

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENCV_NUM_THREADS=1

CMD ["sh", "run.sh"]
