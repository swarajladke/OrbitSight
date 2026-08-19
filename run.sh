#!/bin/sh
set -e

TEAM_NAME="${TEAM_NAME:-orbitsight}"
RUN_DATE="${RUN_DATE:-$(date +%d%m%Y)}"

DATASET_DIR="${ORBITSIGHT_DATASET_DIR:-/OrbitSight_dataset}"
OUTPUT_DIR="${ORBITSIGHT_OUTPUT_DIR:-/work/${TEAM_NAME}/${RUN_DATE}}"

mkdir -p "$OUTPUT_DIR"
echo "dataset: $DATASET_DIR"
echo "output:  $OUTPUT_DIR"

python -m src.infer --input_dir "$DATASET_DIR" --output_dir "$OUTPUT_DIR"
python -m src.validate_predictions --pred-dir "$OUTPUT_DIR"
python -m src.make_report --dataset-dir "$DATASET_DIR" --pred-dir "$OUTPUT_DIR" --out "$OUTPUT_DIR/Evaluation_Metrics.xlsx"

echo "DONE"
