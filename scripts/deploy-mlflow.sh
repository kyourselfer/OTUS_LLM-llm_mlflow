#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/kind-mlflow || true

# Apply with fallback to --validate=false if OpenAPI download/validation fails
kubectl apply -f k8s/mlflow-pv.yaml || kubectl apply --validate=false -f k8s/mlflow-pv.yaml
kubectl apply -f k8s/mlflow.yaml || kubectl apply --validate=false -f k8s/mlflow.yaml

kubectl rollout status deploy/mlflow

