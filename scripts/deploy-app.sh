#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

kubectl rollout status deploy/llm-inference