#!/usr/bin/env bash
set -euo pipefail

if ! command -v kind >/dev/null 2>&1; then
  echo "Please install kind: https://kind.sigs.k8s.io/" && exit 1
fi

if kind get clusters 2>/dev/null | grep -qx llm; then
  echo "kind cluster 'llm' already exists. Skipping creation."
else
  kind create cluster --name llm --config scripts/kind-config.yaml

  # metrics-server (для HPA)
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

  kubectl -n kube-system patch deployment metrics-server \
    --type='json' -p='[
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname"}
  ]' || true
fi

# Ensure kubeconfig is written at a stable path for CI jobs
mkdir -p "$HOME/.kube"
kind get kubeconfig --name llm > "$HOME/.kube/config"
kubectl config use-context kind-llm || true
kubectl cluster-info || true

# Label node(s) for ingress-nginx (kind manifest expects ingress-ready=true)
kubectl get nodes -o name | xargs -I{} kubectl label {} ingress-ready=true --overwrite || true

# Also write kubeconfig to the invoking user's home so shell-runner (root) can use a fixed path
mkdir -p /Users/muse/.kube || true
kind get kubeconfig --name llm > /Users/muse/.kube/config