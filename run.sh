#!/bin/sh
set -e
TEAM_NAME="${TEAM_NAME:-OrbitAI}"
RUN_DATE="${RUN_DATE:-$(date +%d%m%Y)}"
DATASET_DIR="${ORBITSIGHT_DATASET_DIR:-/OrbitSight_dataset}"
OUTPUT_DIR="${ORBITSIGHT_OUTPUT_DIR:-/work/${TEAM_NAME}/${RUN_DATE}}"
mkdir -p "$OUTPUT_DIR"
echo "dataset: $DATASET_DIR"
echo "output:  $OUTPUT_DIR"
python -m src.infer --input_dir "$DATASET_DIR" --output_dir "$OUTPUT_DIR"
python -m src.validate_predictions --pred-dir "$OUTPUT_DIR" || echo "[WARN] validation reported issues"
python -m src.make_report --dataset-dir "$DATASET_DIR" --pred-dir "$OUTPUT_DIR" --out "$OUTPUT_DIR/Evaluation_Metrics.xlsx" || echo "[WARN] metrics report degraded"
for alt in orbitai orbitsight OrbitSight; do
  if [ "$alt" != "$TEAM_NAME" ]; then
    mkdir -p "/work/${alt}/${RUN_DATE}"
    cp -r "$OUTPUT_DIR/." "/work/${alt}/${RUN_DATE}/" || echo "[WARN] mirror to $alt failed"
    echo "mirrored: /work/${alt}/${RUN_DATE}"
  fi
done
echo "DONE"
