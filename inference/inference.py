import os
import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import mlflow
from starlette.responses import Response

# ===== Metrics =====
REQ_COUNTER = Counter(
    "llm_requests_total", "Total LLM requests", ["endpoint"]
)
TOKENS_COUNTER = Counter(
    "llm_tokens_total", "Total tokens generated", ["phase"]  # phase: input|output
)
LATENCY_HIST = Histogram(
    "llm_request_latency_seconds", "LLM request latency (s)", buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)
GPU_MEM_GAUGE = Gauge("llm_gpu_mem_bytes", "Allocated GPU memory (bytes)")
CPU_THREADS_GAUGE = Gauge("llm_cpu_threads", "Torch threads")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    do_sample: bool = True
    stop: Optional[List[str]] = None

class GenerateResponse(BaseModel):
    output: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float

app = FastAPI(title="LLM Inference Service", version="1.0")

MODEL_ID = os.getenv("MODEL_ID", "distilgpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = os.getenv("DTYPE", "float16" if DEVICE == "cuda" else "float32")
TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 4)))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "inference")

# Torch perf tweaks
try:
    torch.set_num_threads(TORCH_THREADS)
    CPU_THREADS_GAUGE.set(TORCH_THREADS)
except Exception:
    pass

if torch.__version__.startswith("2"):
    torch.set_float32_matmul_precision("high")

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=(torch.float16 if DTYPE == "float16" and DEVICE == "cuda" else torch.float32),
    low_cpu_mem_usage=True,
)
model.to(DEVICE)

# Optional: compile (PyTorch 2.x)
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
    except Exception:
        pass

pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}

@app.get("/metrics")
def metrics():
    if DEVICE == "cuda":
        try:
            mem = torch.cuda.memory_allocated()
            GPU_MEM_GAUGE.set(mem)
        except Exception:
            pass
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    REQ_COUNTER.labels(endpoint="/generate").inc()
    start = time.perf_counter()

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_id", MODEL_ID)
            mlflow.log_param("device", DEVICE)
            mlflow.log_param("dtype", DTYPE)
            mlflow.log_params({
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "do_sample": req.do_sample,
            })

            inputs = tokenizer(req.prompt, return_tensors="pt")
            input_tokens = inputs["input_ids"].shape[-1]
            TOKENS_COUNTER.labels(phase="input").inc(input_tokens)

            gen = pipe(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_p=req.top_p,
                eos_token_id=None if not req.stop else [tokenizer.encode(s, add_special_tokens=False)[0] for s in req.stop],
                num_return_sequences=1,
            )
            output_text = gen[0]["generated_text"][len(req.prompt):]

            # Rough token count for output
            output_tokens = len(tokenizer.encode(output_text))
            TOKENS_COUNTER.labels(phase="output").inc(output_tokens)

            latency = time.perf_counter() - start
            LATENCY_HIST.observe(latency)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("input_tokens", input_tokens)
            mlflow.log_metric("output_tokens", output_tokens)
            return GenerateResponse(output=output_text, input_tokens=input_tokens, output_tokens=output_tokens, latency_seconds=latency)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
