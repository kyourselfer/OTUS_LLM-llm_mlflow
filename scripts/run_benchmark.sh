#!/usr/bin/env bash
set -euo pipefail

# Comprehensive LLM Benchmarking Script
# Downloads GLUE, SuperGLUE, SQuAD, LAMBADA, MMLU datasets
# Evaluates models using Perplexity, BLEU, ROUGE, F1, Exact Match

echo "🚀 Starting Comprehensive LLM Benchmark..."

# 1) Ensure cluster and services are up
echo "📋 Checking cluster and services..."
bash scripts/create-kind-cluster.sh
bash scripts/enable-ingress-nginx.sh
bash scripts/deploy-mlflow.sh

# 2) Create venv and install deps
echo "📦 Setting up Python environment..."
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip

# Install required packages
echo "📥 Installing dependencies..."
pip install torch transformers datasets evaluate mlflow numpy nltk rouge_score absl-py

# 3) Ensure MLflow is accessible
echo "🔍 Checking MLflow connectivity..."
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow.localtest.me:8080}

# Wait for MLflow to be ready
echo "⏳ Waiting for MLflow to be ready..."
for i in {1..30}; do
    if curl -s "$MLFLOW_TRACKING_URI" >/dev/null 2>&1; then
        echo "✅ MLflow is ready!"
        break
    fi
    echo "Waiting for MLflow... ($i/30)"
    sleep 2
done

# 4) Run comprehensive benchmark
echo "🧪 Running comprehensive benchmark..."
python scripts/benchmark_datasets.py

# 5) Show results summary
echo "📊 Benchmark completed!"
echo "📈 Check MLflow UI for detailed results: $MLFLOW_TRACKING_URI"
echo "📄 Summary saved to: benchmark_summary.json"

# 6) Optional: Show top models by different metrics
if [ -f "benchmark_summary.json" ]; then
    echo ""
    echo "🏆 Top models by different metrics:"
    echo "=================================="
    
    # Extract and sort by different metrics
    echo "SQuAD F1 Score:"
    jq -r '.[] | select(.squad_f1 != null) | "\(.model_id): \(.squad_f1)"' benchmark_summary.json | sort -k2 -nr | head -3
    
    echo ""
    echo "LAMBADA Perplexity (lower is better):"
    jq -r '.[] | select(.lambada_perplexity != null) | "\(.model_id): \(.lambada_perplexity)"' benchmark_summary.json | sort -k2 -n | head -3
    
    echo ""
    echo "SQuAD Exact Match:"
    jq -r '.[] | select(.squad_exact_match != null) | "\(.model_id): \(.squad_exact_match)"' benchmark_summary.json | sort -k2 -nr | head -3
fi

echo ""
echo "🎉 Benchmark completed successfully!"
echo "💡 To view detailed results, open MLflow UI at: $MLFLOW_TRACKING_URI"
