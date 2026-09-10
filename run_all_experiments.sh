#!/usr/bin/env bash

# Runs the complete comparison experiment suite for:
#   - Deep Sets
#   - Set Transformer (SAB and ISAB with 3 inducing-point settings)
#   - Janossy GRU (1, 5, 10, and 20 inference permutations)
#   - PointNet++ (3 centroid hierarchies)
#
# Every configuration is trained on both datasets with five seeds and 50 epochs.
# Results are stored in a separate directory for every individual run.

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/model_evaluation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
FORCE="${FORCE:-0}"

EPOCHS=50
LEARNING_RATE=0.001
INTERNAL_FEATURES=128
SEEDS=(42 43 44 45 46)

SYNTHETIC_DATASET="${PROJECT_DIR}/data/SyntheticDataset/synthetic_dataset.pt"
CITESEER_DATASET="${PROJECT_DIR}/data/CiteSeer/citeseer_dataset.pt"

DEEPSET_SCRIPT="${PROJECT_DIR}/src/model_code/deepset.py"
SET_TRANSFORMER_SCRIPT="${PROJECT_DIR}/src/model_code/set_transformer.py"
JANOSSY_SCRIPT="${PROJECT_DIR}/src/model_code/janossy_gru.py"
POINTNET_SCRIPT="${PROJECT_DIR}/src/model_code/pointnet_plus.py"

# Set Transformer ISAB comparison.
ISAB_INDUCING_POINTS=(8 16 32)

# PointNet++ comparison
# -> Each entry defines first-level and second-level centroid counts for one hierarchical configuration
POINTNET_FIRST_CENTROIDS=(4 8 16)
POINTNET_SECOND_CENTROIDS=(2 4 8)

JANOSSY_INFERENCE_PERMUTATIONS=(1 5 10 20)

SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
MANIFEST_CSV="${RESULT_ROOT}/run_manifest.csv"

mkdir -p "${RESULT_ROOT}"

if [[ ! -f "${MANIFEST_CSV}" ]]; then
    printf '%s\n' \
        'timestamp;dataset;model;configuration;seed;status;duration_seconds;run_directory' \
        > "${MANIFEST_CSV}"
fi

required_files=(
    "${DEEPSET_SCRIPT}"
    "${SET_TRANSFORMER_SCRIPT}"
    "${JANOSSY_SCRIPT}"
    "${POINTNET_SCRIPT}"
    "${SYNTHETIC_DATASET}"
    "${CITESEER_DATASET}"
)

for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

append_manifest() {
    local dataset_name="$1"
    local model_name="$2"
    local configuration="$3"
    local seed="$4"
    local status="$5"
    local duration="$6"
    local run_directory="$7"

    printf '%s;%s;%s;%s;%s;%s;%s;%s\n' \
        "$(date --iso-8601=seconds)" \
        "${dataset_name}" \
        "${model_name}" \
        "${configuration}" \
        "${seed}" \
        "${status}" \
        "${duration}" \
        "${run_directory}" \
        >> "${MANIFEST_CSV}"
}

extract_metrics() {
    local run_directory="$1"
    local dataset_name="$2"
    local model_name="$3"
    local configuration="$4"
    local seed="$5"

    local checkpoint
    checkpoint="$(find "${run_directory}" -maxdepth 1 -type f -name '*.pt' -print -quit)"

    if [[ -z "${checkpoint}" ]]; then
        echo "No checkpoint found in ${run_directory}" >&2
        return 1
    fi

    "${PYTHON_BIN}" - \
        "${checkpoint}" \
        "${run_directory}/metrics.json" \
        "${SUMMARY_CSV}" \
        "${dataset_name}" \
        "${model_name}" \
        "${configuration}" \
        "${seed}" \
    "${PROJECT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import torch

(
    checkpoint_path,
    metrics_json_path,
    summary_csv_path,
    dataset_name,
    model_name,
    configuration,
    seed,
    project_dir
) = sys.argv[1:]

project_root = Path(project_dir).resolve()
relative_checkpoint_path = (
    Path(checkpoint_path)
    .resolve()
    .relative_to(project_root)
    .as_posix()
)

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

results = dict(checkpoint.get("detailed_results", {}))

# Fallbacks keep collector usable with older checkpoints
for key in ("test_loss", "test_accuracy", "best_epoch", "best_validation_accuracy"):
    if key not in results and key in checkpoint:
        results[key] = checkpoint[key]

metadata = {
    "dataset": dataset_name,
    "model": model_name,
    "configuration": configuration,
    "seed": int(seed),
    "checkpoint": relative_checkpoint_path
}

with open(metrics_json_path, "w", encoding="utf-8") as metrics_file:
    json.dump(
        {"metadata": metadata, "metrics": results},
        metrics_file,
        indent=2,
        ensure_ascii=False
    )

degree_results = results.get("accuracy_by_degree", {})


def degree_value(group, field):
    value = degree_results.get(group, {}).get(field, "")
    return "" if value is None else value


row = {
    "dataset": dataset_name,
    "model": model_name,
    "configuration": configuration,
    "seed": int(seed),
    "test_accuracy": results.get("test_accuracy", ""),
    "macro_f1": results.get("macro_f1", ""),
    "weighted_f1": results.get("weighted_f1", ""),
    "test_loss": results.get("test_loss", ""),
    "best_validation_accuracy": results.get("best_validation_accuracy", ""),
    "best_epoch": results.get("best_epoch", ""),
    "trainable_parameters": results.get("trainable_parameters", ""),
    "training_time_seconds": results.get("training_time_seconds", ""),
    "inference_time_seconds": results.get("inference_time_seconds", ""),
    "inference_time_per_node_ms": results.get(
        "inference_time_per_node_ms", ""
    ),
    "degree_0_accuracy": degree_value("0", "accuracy"),
    "degree_0_count": degree_value("0", "count"),
    "degree_1_2_accuracy": degree_value("1-2", "accuracy"),
    "degree_1_2_count": degree_value("1-2", "count"),
    "degree_3_5_accuracy": degree_value("3-5", "accuracy"),
    "degree_3_5_count": degree_value("3-5", "count"),
    "degree_6_10_accuracy": degree_value("6-10", "accuracy"),
    "degree_6_10_count": degree_value("6-10", "count"),
    "degree_gt_10_accuracy": degree_value(">10", "accuracy"),
    "degree_gt_10_count": degree_value(">10", "count"),
    "precision_per_class": json.dumps(
        results.get("precision_per_class", {}),
        ensure_ascii=False,
        sort_keys=True
    ),
    "recall_per_class": json.dumps(
        results.get("recall_per_class", {}),
        ensure_ascii=False,
        sort_keys=True
    ),
    "checkpoint": relative_checkpoint_path
}

fieldnames = list(row.keys())
summary_path = Path(summary_csv_path)
existing_rows = []

if summary_path.exists():
    with summary_path.open("r", encoding="utf-8", newline="") as input_file:
        existing_rows = list(csv.DictReader(input_file, delimiter=";"))

# Replace an existing row with the same experiment key instead of duplicating it.
key = (dataset_name, model_name, configuration, str(seed))
existing_rows = [
    existing_row
    for existing_row in existing_rows
    if (
        existing_row.get("dataset"),
        existing_row.get("model"),
        existing_row.get("configuration"),
        existing_row.get("seed")
    )
    != key
]
existing_rows.append({name: row.get(name, "") for name in fieldnames})
existing_rows.sort(
    key=lambda item: (
        item.get("dataset", ""),
        item.get("model", ""),
        item.get("configuration", ""),
        int(item.get("seed", 0))
    )
)

with summary_path.open("w", encoding="utf-8", newline="") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames,
        delimiter=";"
    )
    writer.writeheader()
    writer.writerows(existing_rows)
PY
}

run_experiment() {
    local dataset_name="$1"
    local dataset_path="$2"
    local model_name="$3"
    local configuration="$4"
    local seed="$5"
    shift 5
    local command=("$@")

    local run_directory
    run_directory="${RESULT_ROOT}/${dataset_name}/${model_name}/${configuration}/seed_${seed}"
    mkdir -p "${run_directory}"

    if [[ "${FORCE}" != "1" && -f "${run_directory}/SUCCESS" ]]; then
      echo "[SKIP] ${dataset_name} | ${model_name} | ${configuration} | seed=${seed}"
      extract_metrics \
          "${run_directory}" \
          "${dataset_name}" \
          "${model_name}" \
          "${configuration}" \
          "${seed}"
      return $?
  fi

    rm -f "${run_directory}/SUCCESS" "${run_directory}/FAILED"

    {
        printf 'Working directory: %q\n' "${PROJECT_DIR}"
        printf 'Command: '
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "${run_directory}/command.txt"

    echo
    echo "======================================================================"
    echo "Dataset:       ${dataset_name}"
    echo "Model:         ${model_name}"
    echo "Configuration: ${configuration}"
    echo "Seed:          ${seed}"
    echo "Output:        ${run_directory}"
    echo "======================================================================"

    local start_time
    local end_time
    local duration
    local status

    start_time="$(date +%s)"

    (
        cd "${PROJECT_DIR}" || exit 1
        PYTHONUNBUFFERED=1 "${command[@]}"
    ) 2>&1 | tee "${run_directory}/training.log"

    status="${PIPESTATUS[0]}"
    end_time="$(date +%s)"
    duration="$((end_time - start_time))"

    if [[ "${status}" -eq 0 ]]; then
        if extract_metrics \
            "${run_directory}" \
            "${dataset_name}" \
            "${model_name}" \
            "${configuration}" \
            "${seed}"; then
            touch "${run_directory}/SUCCESS"
            append_manifest \
                "${dataset_name}" \
                "${model_name}" \
                "${configuration}" \
                "${seed}" \
                "success" \
                "${duration}" \
                "${run_directory}"
            echo "[SUCCESS] Completed in ${duration} seconds."
            return 0
        fi

        status=90
        echo "Metric extraction failed." >&2
    fi

    printf '%s\n' "${status}" > "${run_directory}/FAILED"
    append_manifest \
        "${dataset_name}" \
        "${model_name}" \
        "${configuration}" \
        "${seed}" \
        "failed_${status}" \
        "${duration}" \
        "${run_directory}"
    echo "[FAILED] Exit status ${status}; see ${run_directory}/training.log" >&2
    return 1
}

run_for_dataset() {
    local dataset_name="$1"
    local dataset_path="$2"
    local failures=0

    for seed in "${SEEDS[@]}"; do
        run_directory="${RESULT_ROOT}/${dataset_name}/deepset/default/seed_${seed}"
        run_experiment \
            "${dataset_name}" \
            "${dataset_path}" \
            "deepset" \
            "default" \
            "${seed}" \
            "${PYTHON_BIN}" "${DEEPSET_SCRIPT}" \
            --dataset-path "${dataset_path}" \
            --internal-features "${INTERNAL_FEATURES}" \
            --epochs "${EPOCHS}" \
            --learning-rate "${LEARNING_RATE}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --output-dir "${run_directory}" \
            || failures=$((failures + 1))
    done

    for seed in "${SEEDS[@]}"; do
        run_directory="${RESULT_ROOT}/${dataset_name}/set_transformer/sab/seed_${seed}"
        run_experiment \
            "${dataset_name}" \
            "${dataset_path}" \
            "set_transformer" \
            "sab" \
            "${seed}" \
            "${PYTHON_BIN}" "${SET_TRANSFORMER_SCRIPT}" \
            --dataset-path "${dataset_path}" \
            --internal-features "${INTERNAL_FEATURES}" \
            --num-heads 4 \
            --num-blocks 2 \
            --attention-block sab \
            --epochs "${EPOCHS}" \
            --learning-rate "${LEARNING_RATE}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --output-dir "${run_directory}" \
            || failures=$((failures + 1))
    done

    for inducing_points in "${ISAB_INDUCING_POINTS[@]}"; do
        configuration="isab_inducing_${inducing_points}"

        for seed in "${SEEDS[@]}"; do
            run_directory="${RESULT_ROOT}/${dataset_name}/set_transformer/${configuration}/seed_${seed}"
            run_experiment \
                "${dataset_name}" \
                "${dataset_path}" \
                "set_transformer" \
                "${configuration}" \
                "${seed}" \
                "${PYTHON_BIN}" "${SET_TRANSFORMER_SCRIPT}" \
                --dataset-path "${dataset_path}" \
                --internal-features "${INTERNAL_FEATURES}" \
                --num-heads 4 \
                --num-blocks 2 \
                --attention-block isab \
                --num-inducing-points "${inducing_points}" \
                --epochs "${EPOCHS}" \
                --learning-rate "${LEARNING_RATE}" \
                --seed "${seed}" \
                --device "${DEVICE}" \
                --output-dir "${run_directory}" \
                || failures=$((failures + 1))
        done
    done

    for permutations in "${JANOSSY_INFERENCE_PERMUTATIONS[@]}"; do
        configuration="inference_permutations_${permutations}"

        for seed in "${SEEDS[@]}"; do
            run_directory="${RESULT_ROOT}/${dataset_name}/janossy_gru/${configuration}/seed_${seed}"
            run_experiment \
                "${dataset_name}" \
                "${dataset_path}" \
                "janossy_gru" \
                "${configuration}" \
                "${seed}" \
                "${PYTHON_BIN}" "${JANOSSY_SCRIPT}" \
                --dataset-path "${dataset_path}" \
                --internal-features "${INTERNAL_FEATURES}" \
                --inference-permutations "${permutations}" \
                --epochs "${EPOCHS}" \
                --learning-rate "${LEARNING_RATE}" \
                --seed "${seed}" \
                --device "${DEVICE}" \
                --output-dir "${run_directory}" \
                || failures=$((failures + 1))
        done
    done

    for index in "${!POINTNET_FIRST_CENTROIDS[@]}"; do
        first_centroids="${POINTNET_FIRST_CENTROIDS[index]}"
        second_centroids="${POINTNET_SECOND_CENTROIDS[index]}"
        configuration="centroids_${first_centroids}_${second_centroids}"

        for seed in "${SEEDS[@]}"; do
            run_directory="${RESULT_ROOT}/${dataset_name}/pointnet_plus/${configuration}/seed_${seed}"
            run_experiment \
                "${dataset_name}" \
                "${dataset_path}" \
                "pointnet_plus" \
                "${configuration}" \
                "${seed}" \
                "${PYTHON_BIN}" "${POINTNET_SCRIPT}" \
                --dataset-path "${dataset_path}" \
                --internal-features "${INTERNAL_FEATURES}" \
                --coordinate-features 16 \
                --first-centroids "${first_centroids}" \
                --first-neighbors 8 \
                --second-centroids "${second_centroids}" \
                --second-neighbors 4 \
                --epochs "${EPOCHS}" \
                --learning-rate "${LEARNING_RATE}" \
                --seed "${seed}" \
                --device "${DEVICE}" \
                --output-dir "${run_directory}" \
                || failures=$((failures + 1))
        done
    done

    return "${failures}"
}

echo "Project directory: ${PROJECT_DIR}"
echo "Result directory:  ${RESULT_ROOT}"
echo "Device setting:    ${DEVICE}"
echo "Epochs per run:    ${EPOCHS}"
echo "Seeds:             ${SEEDS[*]}"
echo "Planned runs:      120"
echo

TOTAL_FAILURES=0

run_for_dataset "synthetic" "${SYNTHETIC_DATASET}" || TOTAL_FAILURES=$((TOTAL_FAILURES + $?))
run_for_dataset "citeseer" "${CITESEER_DATASET}" || TOTAL_FAILURES=$((TOTAL_FAILURES + $?))

echo
echo "======================================================================"
echo "Experiment suite finished."
echo "Summary:  ${SUMMARY_CSV}"
echo "Manifest: ${MANIFEST_CSV}"
echo "Failures: ${TOTAL_FAILURES}"
echo "======================================================================"

if [[ "${TOTAL_FAILURES}" -ne 0 ]]; then
    exit 1
fi
