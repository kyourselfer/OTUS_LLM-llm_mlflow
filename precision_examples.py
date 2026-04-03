#!/usr/bin/env python
"""
Precision & Quantization Examples (PyTorch, single file)

python precision_examples.py --mode fp16

Covers:
  1) FP16 mixed precision (autocast + GradScaler)
  2) BF16 mixed precision (no scaler)
  3) FP8 (H100) via NVIDIA TransformerEngine (optional)
  4) INT8 / INT4 inference with bitsandbytes (Transformers)
  5) KV-cache quantization (INT8) toy implementation

Notes:
- Examples are minimal and self-contained where possible.
- Sections (3) and (4) require optional libs (transformer_engine, bitsandbytes, transformers).
  Code guards with try/except and prints helpful hints if unavailable.
- No backprop in quantized inference sections; focus is on setup & usage.
"""

import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------
# Toy model and data
# ----------------------------------------------

class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def toy_batch(bs=32, d=1024, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(bs, d, device=device)
    y = torch.randn(bs, d, device=device)
    return x, y


# ----------------------------------------------
# 1) FP16 mixed precision training
# ----------------------------------------------

def demo_fp16_mixed_precision(steps=10):
    """FP16 on Tensor Cores (A100/H100). Use GradScaler to avoid overflow."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for step in range(steps):
        x, y = toy_batch(device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda', dtype=torch.float16):
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 2 == 0:
            print(f"[FP16] step={step} loss={loss.item():.4f}")


# ----------------------------------------------
# 2) BF16 mixed precision training
# ----------------------------------------------

def demo_bf16_mixed_precision(steps=10):
    """BF16: wider exponent, typically stable; no GradScaler needed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    autocast_ok = (device.type == 'cuda')

    for step in range(steps):
        x, y = toy_batch(device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=autocast_ok, dtype=torch.bfloat16):
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
        loss.backward()
        opt.step()
        if step % 2 == 0:
            print(f"[BF16] step={step} loss={loss.item():.4f}")


# ----------------------------------------------
# 3) FP8 with TransformerEngine (optional, H100)
# ----------------------------------------------

def demo_fp8_transformer_engine(steps=5):
    """FP8 demo using NVIDIA TransformerEngine. Requires H100 + 'transformer_engine'.
    Accumulations typically in FP16/BF16; FP8 E4M3/E5M2 formats for activations/weights.
    """
    try:
        import transformer_engine.pytorch as te
    except Exception as e:
        print("[FP8] transformer_engine not available. Install: pip install transformer-engine")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('[FP8] CUDA device not found.')
        return

    torch.manual_seed(0)

    # TE Linear layers with FP8 autocast
    model = nn.Sequential(
        te.Linear(1024, 4096),
        nn.GELU(),
        te.Linear(4096, 1024),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    from transformer_engine.pytorch.fp8 import is_fp8_available
    if not is_fp8_available():
        print('[FP8] FP8 not available on this hardware/runtime.')
        return

    for step in range(steps):
        x, y = toy_batch(device=device)
        opt.zero_grad(set_to_none=True)
        with te.fp8_autocast(enabled=True):  # scales and formats handled internally
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
        loss.backward()
        opt.step()
        print(f"[FP8] step={step} loss={loss.item():.4f}")


# ----------------------------------------------
# 4) INT8 / INT4 inference (bitsandbytes + Transformers)
# ----------------------------------------------

def demo_transformers_int8_int4(prompt: str = "Hello", four_bit: bool = False):
    """Load a model in 8-bit or 4-bit and generate text.
    Requires: transformers>=4.30, bitsandbytes.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import bitsandbytes as bnb  # noqa: F401
    except Exception:
        print('[INTx] Need transformers and bitsandbytes installed.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = os.environ.get('DEMO_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
    print(f"[INTx] Loading {model_name} in {'4-bit' if four_bit else '8-bit'}…")

    if four_bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
            bnb_4bit_quant_type='nf4',  # good accuracy vs memory
            bnb_4bit_use_double_quant=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            load_in_8bit=True,
        )

    tok = AutoTokenizer.from_pretrained(model_name)
    inputs = tok(prompt, return_tensors='pt').to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32)
    print('[INTx] Output:\n', tok.decode(out[0], skip_special_tokens=True))


# ----------------------------------------------
# 5) KV-cache quantization (toy INT8 with per-head scales)
# ----------------------------------------------

@dataclass
class KVCache8:
    K_q: torch.Tensor  # int8 [B, n_heads, T, H_head]
    V_q: torch.Tensor  # int8 [B, n_heads, T, H_head]
    sK: torch.Tensor   # scales float32 [B, n_heads, 1, 1]
    sV: torch.Tensor   # scales float32 [B, n_heads, 1, 1]


def quantize_per_head_int8(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to int8 with per-(B,head) symmetric scales. x: [B, H, T, D] or [B, H, D]…
    Here expect x: [B, n_heads, T, H_head]. Return (q, scale_per_bh).
    """
    # max abs per (B,head)
    max_abs = x.abs().amax(dim=(-1, -2), keepdim=True)  # [B,H,1,1]
    scale = (max_abs / 127.0).clamp(min=eps)
    q = torch.round((x / scale).clamp(-127, 127)).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def build_kv_cache_int8(K: torch.Tensor, V: torch.Tensor) -> KVCache8:
    """K,V float32/bf16/… with shape [B, n_heads, T, H_head] -> int8 cache with per-head scales."""
    K_q, sK = quantize_per_head_int8(K)
    V_q, sV = quantize_per_head_int8(V)
    return KVCache8(K_q=K_q, V_q=V_q, sK=sK, sV=sV)


def attention_with_int8_kv(Q: torch.Tensor, cache: KVCache8) -> torch.Tensor:
    """Compute attention using dequantized K/V on the fly.
    Q: [B, H, Tq, Dh]; cache.K_q/V_q: [B, H, Tk, Dh]
    """
    B, H, Tq, Dh = Q.shape
    K = dequantize_int8(cache.K_q, cache.sK)
    V = dequantize_int8(cache.V_q, cache.sV)
    att = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Dh)
    prob = att.softmax(dim=-1)
    out = torch.matmul(prob, V)
    return out


def demo_kv_cache_quant():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    B, H, Tk, Tq, Dh = 2, 8, 512, 16, 64
    K = torch.randn(B, H, Tk, Dh, device=device)
    V = torch.randn(B, H, Tk, Dh, device=device)
    Q = torch.randn(B, H, Tq, Dh, device=device)

    cache8 = build_kv_cache_int8(K, V)
    out8 = attention_with_int8_kv(Q, cache8)

    # Reference FP32
    att = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Dh)
    prob = att.softmax(dim=-1)
    out_ref = torch.matmul(prob, V)

    err = (out8 - out_ref).abs().mean().item()
    print(f"[KV INT8] mean abs error vs FP32: {err:.6f}")


# ----------------------------------------------
# CLI
# ----------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, required=True,
                   choices=['fp16', 'bf16', 'fp8', 'int8', 'int4', 'kv8'],
                   help='Which demo to run')
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--prompt', type=str, default='Hello from quantized model!')
    args = p.parse_args()

    if args.mode == 'fp16':
        demo_fp16_mixed_precision(steps=args.steps)
    elif args.mode == 'bf16':
        demo_bf16_mixed_precision(steps=args.steps)
    elif args.mode == 'fp8':
        demo_fp8_transformer_engine(steps=args.steps)
    elif args.mode == 'int8':
        demo_transformers_int8_int4(prompt=args.prompt, four_bit=False)
    elif args.mode == 'int4':
        demo_transformers_int8_int4(prompt=args.prompt, four_bit=True)
    elif args.mode == 'kv8':
        demo_kv_cache_quant()
    else:
        raise SystemExit('Unknown mode')
