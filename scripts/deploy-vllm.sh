#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/vllm-deployment.yaml

kubectl rollout status deploy/vllm

