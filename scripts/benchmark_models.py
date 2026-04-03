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
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return text.strip()


def evaluate_model(model_id: str):
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
        p_tokens = normalize_text(pred)
        r_tokens = normalize_text(ex["answer"])
        # token_f1
        overlap = 0
        r_counts = {}
        for t in r_tokens:
            r_counts[t] = r_counts.get(t, 0) + 1
        for t in p_tokens:
            if r_counts.get(t, 0) > 0:
                overlap += 1
                r_counts[t] -= 1
        precision = overlap / len(p_tokens) if p_tokens else 0.0
        recall = overlap / len(r_tokens) if r_tokens else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
        em_scores.append(1.0 if p_tokens == r_tokens else 0.0)

    metrics = {
        "f1_mean": float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
        "em_mean": float(sum(em_scores) / len(em_scores)) if em_scores else 0.0,
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



