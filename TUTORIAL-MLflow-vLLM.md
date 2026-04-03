## Tutorial: Compare HF models with MLflow, pick the best, and serve via vLLM

This tutorial shows how to:
- Evaluate several small Hugging Face causal LMs on a tiny QA set
- Log metrics and artifacts to MLflow
- Pick the best model by score
- Update the vLLM Deployment to serve the chosen model
- Query vLLM with a QA example

Prereqs:
- Cluster and services from this repo are running (see README)
- MLflow UI reachable at `http://mlflow.localtest.me:8080`
- vLLM reachable at `http://vllm.localtest.me:8080/v1`
- Python 3.10/3.11 on your host

### 1) Install local deps (one-time)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install transformers torch mlflow datasets evaluate
```

### 2) Create a tiny QA eval set
We will use a few handcrafted samples for speed. You can swap this with SQuAD samples later.

```python
EVAL_SAMPLES = [
    {
        "context": "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.",
        "question": "What does Kubernetes automate?",
        "answer": "deployment, scaling, and management of containerized applications",
    },
    {
        "context": "vLLM is a high-throughput and memory-efficient inference and serving engine for large language models.",
        "question": "What is vLLM designed for?",
        "answer": "inference and serving engine for large language models",
    },
    {
        "context": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
        "question": "What does MLflow manage?",
        "answer": "the end-to-end machine learning lifecycle",
    },
]
```

### 3) Benchmark several HF models and log to MLflow
The script below:
- Compares multiple small causal LMs using a simple prompt for QA
- Computes a token-level F1 and Exact Match (EM)
- Logs params/metrics/artifacts to MLflow
- Prints the best model

Save as `scripts/benchmark_models.py` and run.

```python
import os
import json
from typing import List, Dict

import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


EVAL_SAMPLES: List[Dict[str, str]] = [
    {
        "context": "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.",
        "question": "What does Kubernetes automate?",
        "answer": "deployment, scaling, and management of containerized applications",
    },
    {
        "context": "vLLM is a high-throughput and memory-efficient inference and serving engine for large language models.",
        "question": "What is vLLM designed for?",
        "answer": "inference and serving engine for large language models",
    },
    {
        "context": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
        "question": "What does MLflow manage?",
        "answer": "the end-to-end machine learning lifecycle",
    },
]

MODELS = [
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125M",
    "gpt2",
]


def normalize_text(s: str) -> List[str]:
    return s.strip().lower().replace("\n", " ").split()


def token_f1(pred: str, ref: str) -> float:
    p = normalize_text(pred)
    r = normalize_text(ref)
    if not p or not r:
        return 0.0
    overlap = 0
    r_counts = {}
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1
    for t in p:
        if r_counts.get(t, 0) > 0:
            overlap += 1
            r_counts[t] -= 1
    precision = overlap / len(p)
    recall = overlap / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def build_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful assistant. Answer succinctly based only on the context.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Heuristic: take text after the last "Answer:" occurrence
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return text.strip()


def evaluate_model(model_id: str) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)

    preds = []
    f1_scores = []
    em_scores = []
    for ex in EVAL_SAMPLES:
        prompt = build_prompt(ex["context"], ex["question"])
        pred = generate_answer(tokenizer, model, prompt)
        preds.append({"prompt": prompt, "pred": pred, "ref": ex["answer"]})
        f1 = token_f1(pred, ex["answer"])
        em = exact_match(pred, ex["answer"])
        f1_scores.append(f1)
        em_scores.append(em)

    metrics = {
        "f1_mean": float(sum(f1_scores) / len(f1_scores)),
        "em_mean": float(sum(em_scores) / len(em_scores)),
    }
    return metrics, preds


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.localtest.me:8080")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("model-comparison")

    best_model = None
    best_score = -1.0

    for model_id in MODELS:
        with mlflow.start_run(run_name=model_id):
            mlflow.log_param("model_id", model_id)
            (metrics, preds) = evaluate_model(model_id)
            mlflow.log_metrics(metrics)
            with open("predictions.json", "w") as f:
                json.dump(preds, f, indent=2)
            mlflow.log_artifact("predictions.json")
            os.remove("predictions.json")

            print(f"Model {model_id} → F1={metrics['f1_mean']:.4f}, EM={metrics['em_mean']:.4f}")
            if metrics["f1_mean"] > best_score:
                best_score = metrics["f1_mean"]
                best_model = model_id

    print("\nBest model by F1:", best_model, f"(F1={best_score:.4f})")
    with open("best_model.txt", "w") as f:
        f.write(best_model)
    print("Saved best model to best_model.txt")


if __name__ == "__main__":
    main()
```

Run it:
```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow.localtest.me:8080}
python scripts/benchmark_models.py
```

Open MLflow UI and compare runs: `http://mlflow.localtest.me:8080`

### 4) Update vLLM to serve the best model
The vLLM Deployment in this repo serves one model at a time via env `MODEL_ID`.

```bash
BEST=$(cat best_model.txt)
kubectl set env deploy/vllm MODEL_ID="$BEST"
kubectl rollout status deploy/vllm
kubectl get pods -l app=vllm
```

If you deploy via CI, set `MODEL_ID` in `k8s/vllm-deployment.yaml` or patch it similarly in your deploy step.

### 5) Query vLLM for QA via OpenAI-compatible API
```bash
HOST=${HOST:-http://vllm.localtest.me:8080}
PROMPT='You are a helpful assistant. Answer succinctly based only on the context.
Context: MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.
Question: What does MLflow manage?
Answer:'

curl -s -X POST "$HOST/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$BEST\", \"prompt\": \"${PROMPT//\n/\\n}\", \"max_tokens\": 64, \"temperature\": 0.7, \"top_p\": 0.9}" | jq .choices[0].text
```

Tip: if you prefer the Chat Completions format and your model supports system/user formatting, call `/v1/chat/completions` instead.

### Notes
- Small models (125M) are used to keep CPU inference quick. Quality will be modest; swap in better instruct-tuned models if you have resources.
- Metrics here are simple (token F1/EM); for rigorous evaluation use `evaluate` and real QA datasets like SQuAD and proper prompt templates.
- Ensure MLflow is reachable from your host. For k8s kind setup in this repo, MLflow is at `http://mlflow.localtest.me:8080` via Ingress.

