#!/usr/bin/env bash
set -euo pipefail

# Загрузка базового образа Triton и тегирование локального имени для kind
# Базовый образ: ghcr.io/triton-inference-server/server:23.10-py3 (CPU)
# Итоговый тег: triton-server:local

IMAGE_BASE=${IMAGE_BASE:-ghcr.io/triton-inference-server/server:23.10-py3}
IMAGE_TAG=${IMAGE_TAG:-triton-server:local}
KIND_CLUSTER_NAME=${KIND_CLUSTER_NAME:-llm}

echo "Pulling base image $IMAGE_BASE..."
docker pull "$IMAGE_BASE"

echo "Tagging image as $IMAGE_TAG..."
docker tag "$IMAGE_BASE" "$IMAGE_TAG"

echo "Loading image into kind cluster $KIND_CLUSTER_NAME..."
kind load docker-image "$IMAGE_TAG" --name "$KIND_CLUSTER_NAME"

echo "Done. Set image to $IMAGE_TAG and imagePullPolicy IfNotPresent in your Deployment."