#!/usr/bin/env bash
set -euo pipefail

VERSION=${VERSION:-controller-v1.11.2}

# Иногда валидация OpenAPI может падать, поэтому делаем попытку без валидации при ошибке
if ! kubectl apply -f "https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/kind/deploy.yaml"; then
  echo "Retrying ingress-nginx apply with --validate=false..."
  kubectl apply --validate=false -f "https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/kind/deploy.yaml"
fi

# Ждём готовности контроллера
kubectl wait --namespace ingress-nginx \
  --for=condition=Ready pods \
  --selector=app.kubernetes.io/component=controller \
  --timeout=180s