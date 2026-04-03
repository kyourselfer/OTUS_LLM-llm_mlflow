#!/usr/bin/env bash
set -euo pipefail

scripts/build-and-load-triton.sh

# Use local image inside kind
kubectl apply -f k8s/triton-deployment-kind.yaml
kubectl apply -f k8s/triton-service.yaml
kubectl apply -f k8s/triton-ingress.yaml

kubectl rollout status deploy/triton-server

