#!/usr/bin/env bash
set -euo pipefail
#
## 1) Ensure cluster and services are up
#bash scripts/create-kind-cluster.sh
#bash scripts/enable-ingress-nginx.sh
#bash scripts/deploy-mlflow.sh
#bash scripts/deploy-vllm.sh

# 2) Create venv and install deps locally (host)
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
pip install transformers torch mlflow datasets evaluate

# 3) Run benchmark and pick best model
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow.localtest.me:8080}
python scripts/benchmark_models.py

bash scripts/run_benchmark.sh



