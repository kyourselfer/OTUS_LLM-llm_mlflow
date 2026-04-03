#!/usr/bin/env bash
set -euo pipefail

# Если не используешь ingress, просто пробрось сервис на localhost:8080
kubectl port-forward svc/llm-inference 8080:80