#!/usr/bin/env bash
set -euo pipefail

IMAGE=llm-ops-workshop:latest

docker build -t $IMAGE -f inference/Dockerfile .

# Загрузить образ в kind
kind load docker-image $IMAGE --name llm