#!/usr/bin/env bash
set -euo pipefail

# Optionally skip kind cluster prep (CI does this in a separate stage)
if [[ "${SKIP_KIND_PREP:-}" != "1" ]]; then
  bash scripts/create-kind-cluster.sh
  export KUBECONFIG="$HOME/.kube/config"
  kubectl config use-context kind-llm || true
  for i in {1..30}; do kubectl cluster-info && break || sleep 2; done
fi

bash scripts/deploy-mlflow.sh
bash scripts/deploy-vllm.sh
bash scripts/deploy-triton.sh
bash scripts/deploy-app.sh

