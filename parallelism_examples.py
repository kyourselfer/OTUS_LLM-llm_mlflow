#!/usr/bin/env python
"""
Usage:
  # 1) Data/batching (single GPU/CPU is fine)
  python parallelism_examples.py --example data

  # 2) Tensor (intra-layer) parallelism  (run with >=2 processes)
  torchrun --standalone --nproc_per_node=2 parallelism_examples.py --example tensor

  # 3) Pipeline (inter-layer) parallelism (ideally 2 GPUs; falls back to CPU)
  python parallelism_examples.py --example pipeline --micro_batches 8

  # 4) Sequence/context parallelism (run with >=2 processes)
  torchrun --standalone --nproc_per_node=2 parallelism_examples.py --example sequence

  # 5) Expert/MoE parallelism (run with >=2 processes)
  torchrun --standalone --nproc_per_node=2 parallelism_examples.py --example moe

Notes:
- Examples 2/4/5 require torch.distributed initialized across multiple processes.
- Backend will use 'nccl' if CUDA is available, otherwise 'gloo'.
- These are educational demos (no backprop). For real training see Megatron-LM/DeepSpeed/FSDP.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# -----------------------------
# Utilities
# -----------------------------

def has_cuda() -> bool:
    return torch.cuda.is_available()


def dev(i: int = 0) -> torch.device:
    if has_cuda():
        return torch.device(f"cuda:{min(i, torch.cuda.device_count()-1)}")
    return torch.device("cpu")


def setup_dist_if_needed():
    """Initialize torch.distributed if environment suggests multi-proc run.
    Safe to call multiple times.
    """
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if has_cuda() else "gloo"
        # torchrun sets env vars: RANK, WORLD_SIZE, LOCAL_RANK, etc.
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend)
        else:
            # Single-process fallback (no-op distributed)
            pass


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def print0(*args, **kwargs):
    if rank() == 0:
        print(*args, **kwargs)


# -----------------------------
# 1) Data / Batching example
# -----------------------------

class TinyLM(nn.Module):
    def __init__(self, d=256, vocab=4096, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True),
            num_layers=n_layers,
        )
        self.out = nn.Linear(d, vocab)

    def forward(self, x, pad_mask=None):
        h = self.emb(x)
        h = self.lm(h, src_key_padding_mask=pad_mask)
        return self.out(h)


def run_data_batching():
    print0("[Data] batching demo…")
    torch.manual_seed(0)
    model = TinyLM().to(dev())

    # three prompts with different lengths
    prompts = [
        torch.randint(5, 4000, (23,), dtype=torch.long),
        torch.randint(5, 4000, (31,), dtype=torch.long),
        torch.randint(5, 4000, (12,), dtype=torch.long),
    ]

    pad_id = 0
    max_len = max(p.size(0) for p in prompts)
    batch = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    pad_mask = torch.ones((len(prompts), max_len), dtype=torch.bool)  # True=PAD
    for i, p in enumerate(prompts):
        batch[i, : p.size(0)] = p
        pad_mask[i, : p.size(0)] = False

    batch = batch.to(dev())
    pad_mask = pad_mask.to(dev())

    with torch.inference_mode():
        logits = model(batch, pad_mask)
    print0("[Data] logits shape:", tuple(logits.shape))  # [B, T, V]


# -----------------------------
# 2) Tensor (intra-layer) parallelism
# -----------------------------

class ColumnParallelLinear(nn.Module):
    """Shard W[in, out] across columns (out dimension)."""

    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()
        ws = world_size()
        assert out_features % ws == 0, "out_features must be divisible by world_size"
        self.out_per_rank = out_features // ws
        self.weight = nn.Parameter(torch.empty(in_features, self.out_per_rank))
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None
        nn.init.xavier_normal_(self.weight)
        self.gather_output = gather_output

    def forward(self, x):  # x: [B, T, in]
        y_local = x.matmul(self.weight)  # [B, T, out/WS]
        if self.bias is not None:
            y_local = y_local + self.bias
        if world_size() > 1 and self.gather_output:
            y_list = [torch.empty_like(y_local) for _ in range(world_size())]
            dist.all_gather(y_list, y_local)
            return torch.cat(y_list, dim=-1)
        return y_local


class RowParallelLinear(nn.Module):
    """Shard W[in, out] across rows (in dimension). Input must be sharded accordingly."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        ws = world_size()
        assert in_features % ws == 0, "in_features must be divisible by world_size"
        self.in_per_rank = in_features // ws
        self.weight = nn.Parameter(torch.empty(self.in_per_rank, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_normal_(self.weight)

    def forward(self, x_shard):  # x_shard: [B, T, in/WS]
        y_partial = x_shard.matmul(self.weight)  # [B, T, out]
        if world_size() > 1:
            dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            y_partial = y_partial + self.bias
        return y_partial


class TPMLP(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_hidden, gather_output=False)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(d_hidden, d_model)

    def forward(self, x):  # x: [B, T, d_model]
        h_shard = self.fc1(x)  # [B, T, d_hidden/WS]
        h_shard = self.act(h_shard)
        out = self.fc2(h_shard)  # [B, T, d_model]
        return out


def run_tensor_parallel():
    setup_dist_if_needed()
    torch.manual_seed(0 + rank())

    d_model = 512
    d_hidden = 2048
    assert d_hidden % max(world_size(), 1) == 0
    assert d_model % max(world_size(), 1) == 0 or True  # only needed for RowParallel in_features

    x = torch.randn(8, 16, d_model, device=dev())
    model = TPMLP(d_model, d_hidden).to(dev())
    # simple forward
    with torch.inference_mode():
        y = model(x)
    print0(f"[Tensor] output shape: {tuple(y.shape)} | world_size={world_size()}")

    cleanup_dist()


# -----------------------------
# 3) Pipeline (inter-layer) parallelism
# -----------------------------

def run_pipeline(micro_batches: int = 4):
    print0("[Pipeline] demo…")
    d0 = dev(0)
    d1 = dev(1) if has_cuda() and torch.cuda.device_count() >= 2 else dev(0)
    same_device = d0 == d1
    if same_device:
        print0("[Pipeline] Only one device detected — simulating pipeline on a single device.")

    part0 = nn.Sequential(
        nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 2048)
    ).to(d0)
    part1 = nn.Sequential(nn.ReLU(), nn.Linear(2048, 1024)).to(d1)

    x = torch.randn(64, 1024, device=d0)
    chunks = x.chunk(micro_batches, dim=0)

    h_queue: List[torch.Tensor] = []
    for xb in chunks:
        h = part0(xb.to(d0))
        h = h.to(d1, non_blocking=True)
        h_queue.append(h)

    torch.cuda.synchronize(d1) if has_cuda() else None
    out_chunks = [part1(h) for h in h_queue]
    y = torch.cat(out_chunks, dim=0).to(d0)

    print0("[Pipeline] out shape:", tuple(y.shape), "| micro_batches=", micro_batches)


# -----------------------------
# 4) Sequence / Context parallelism
# -----------------------------

def seq_parallel_attention(Q_local, K_local, V_local) -> torch.Tensor:
    """All-gather K/V across sequence shards, local attention for Q_local.
    Shapes: [B, T_local, H]
    """
    ws = world_size()
    if ws == 1:
        K_full = K_local
        V_full = V_local
    else:
        K_list = [torch.empty_like(K_local) for _ in range(ws)]
        V_list = [torch.empty_like(V_local) for _ in range(ws)]
        dist.all_gather(K_list, K_local)
        dist.all_gather(V_list, V_local)
        K_full = torch.cat(K_list, dim=1)
        V_full = torch.cat(V_list, dim=1)

    scale = (Q_local.size(-1) ** 0.5)
    attn = torch.matmul(Q_local, K_full.transpose(-2, -1)) / scale
    probs = F.softmax(attn, dim=-1)
    out = torch.matmul(probs, V_full)
    return out


def run_sequence_parallel():
    setup_dist_if_needed()
    torch.manual_seed(1234 + rank())

    B, T_total, H = 2, 128, 64
    ws = world_size()
    # simple equal split; last shard takes the remainder
    base = T_total // ws
    extra = T_total % ws
    T_local = base + (1 if rank() < extra else 0)

    Q_local = torch.randn(B, T_local, H, device=dev())
    K_local = torch.randn(B, T_local, H, device=dev())
    V_local = torch.randn(B, T_local, H, device=dev())

    with torch.inference_mode():
        O_local = seq_parallel_attention(Q_local, K_local, V_local)
    print0(f"[Seq] rank={rank()} local_out shape={tuple(O_local.shape)} | world_size={ws}")

    cleanup_dist()


# -----------------------------
# 5) Expert / MoE parallelism (top-1) with round-trip routing
# -----------------------------

class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@dataclass
class Packet:
    data: torch.Tensor   # [N, H]
    idx: torch.Tensor    # [N] original flat indices


def all_to_all_packets(send_packets: List[Packet]) -> List[Packet]:
    """Perform all_to_all_single for tokens (2 tensors: data, idx), with split sizes."""
    ws = world_size()
    device = send_packets[0].data.device if send_packets else dev()

    send_sizes = torch.tensor([p.data.size(0) for p in send_packets], device=device, dtype=torch.int64)
    max_send = int(send_sizes.max().item()) if ws > 0 else 0

    # Pad each packet to max_send
    def pad_to(t: torch.Tensor, n: int, pad_dim: int = 0):
        if t.size(0) == n:
            return t
        pad_shape = list(t.shape)
        pad_shape[0] = n - t.size(0)
        return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=0)

    send_data = torch.stack([pad_to(p.data, max_send) for p in send_packets], dim=0)  # [ws, max_send, H]
    send_idx  = torch.stack([pad_to(p.idx,  max_send) for p in send_packets], dim=0)   # [ws, max_send]

    flat_send_data = send_data.reshape(ws * max_send, -1).contiguous()
    flat_send_idx  = send_idx.reshape(ws * max_send).contiguous()

    recv_data = torch.empty_like(flat_send_data)
    recv_idx  = torch.empty_like(flat_send_idx)

    # Perform paired all_to_all_single (sizes symmetric for simplicity)
    dist.all_to_all_single(recv_data, flat_send_data)
    dist.all_to_all_single(recv_idx, flat_send_idx)

    # Reshape back and trim paddings by *received* sizes (mirror of send_sizes from peers)
    recv_data = recv_data.reshape(ws, max_send, -1)
    recv_idx  = recv_idx.reshape(ws, max_send)

    # We don't know exact recv sizes per sender without extra exchange; assume symmetry here.
    # For a demo, drop zero rows using idx!=0 (but 0 can be a valid index). To avoid ambiguity, we
    # shift all sent idx by +1 and then subtract 1 after receiving.
    recv_idx = recv_idx.long()
    valid_mask = recv_idx > 0
    recv_packets: List[Packet] = []
    for src in range(ws):
        mask = valid_mask[src]
        data_part = recv_data[src][mask]
        idx_part = recv_idx[src][mask] - 1
        recv_packets.append(Packet(data_part, idx_part))
    return recv_packets


def run_moe_parallel():
    setup_dist_if_needed()
    torch.manual_seed(2024 + rank())

    ws = world_size()
    if ws < 2:
        print0("[MoE] Need world_size>=2. Run with torchrun … --nproc_per_node=2")
        cleanup_dist()
        return

    B, T, H = 4, 32, 64
    d_hidden = 128

    # One expert per rank
    expert = Expert(H, d_hidden).to(dev())
    router = nn.Linear(H, ws, bias=False).to(dev())

    x = torch.randn(B, T, H, device=dev())
    flat_x = x.reshape(-1, H)  # [N, H]
    N = flat_x.size(0)

    # Top-1 routing
    with torch.inference_mode():
        logits = router(flat_x)                 # [N, ws]
        target_rank = logits.argmax(dim=-1)     # [N]

    # Build send packets per target rank, carrying +1-shifted indices for padding-safe mask
    send_packets: List[Packet] = []
    for r in range(ws):
        mask = target_rank == r
        data_r = flat_x[mask]
        idx_r = torch.nonzero(mask, as_tuple=False).flatten() + 1  # +1 shift
        if data_r.numel() == 0:
            data_r = torch.zeros((0, H), device=dev())
            idx_r = torch.zeros((0,), device=dev(), dtype=torch.long)
        send_packets.append(Packet(data_r, idx_r))

    # Dispatch to experts (each rank receives tokens destined to it)
    recv_packets = all_to_all_packets(send_packets)
    # Combine packets from all senders for my expert
    if len(recv_packets) > 0:
        my_tokens = torch.cat([p.data for p in recv_packets], dim=0)
        my_indices = torch.cat([p.idx for p in recv_packets], dim=0)
    else:
        my_tokens = torch.zeros((0, H), device=dev())
        my_indices = torch.zeros((0,), device=dev(), dtype=torch.long)

    # Process by my expert
    with torch.inference_mode():
        my_out = expert(my_tokens)

    # Prepare return packets: send results back to original owners
    # For symmetry, we need to split my_out by original sender ranks.
    # We don't explicitly track sender here; for a demo we'll just broadcast back to all and let
    # non-owners drop (using index ownership range per rank). We'll compute owner by range:
    # Each rank originally owned flat indices in [start:end) where start is chunked equally.
    # This holds because we formed indices by torch.nonzero over flat_x in this single process.

    # Compute owner by chunking N over ws
    base = N // ws
    extra = N % ws
    def owner_of(idx: torch.Tensor) -> torch.Tensor:
        owners = torch.empty_like(idx)
        # Build boundaries
        bounds: List[Tuple[int, int]] = []
        s = 0
        for r in range(ws):
            length = base + (1 if r < extra else 0)
            bounds.append((s, s + length))
            s += length
        for r, (lo, hi) in enumerate(bounds):
            mask = (idx >= lo) & (idx < hi)
            owners[mask] = r
        return owners

    my_owners = owner_of(my_indices)

    # Build send packets per destination owner
    ret_packets: List[Packet] = []
    for r in range(ws):
        mask = (my_owners == r)
        ret_data = my_out[mask]
        ret_idx = my_indices[mask] + 1  # +1 for safe mask in all_to_all
        if ret_data.numel() == 0:
            ret_data = torch.zeros((0, H), device=dev())
            ret_idx = torch.zeros((0,), device=dev(), dtype=torch.long)
        ret_packets.append(Packet(ret_data, ret_idx))

    # Return to owners
    back_packets = all_to_all_packets(ret_packets)

    # Stitch results in original order on each rank (only keep my portion [start:end))
    start = sum(base + (1 if r < extra else 0) for r in range(rank()))
    end = start + base + (1 if rank() < extra else 0)
    myN = end - start

    if len(back_packets) > 0:
        back_data = torch.cat([p.data for p in back_packets], dim=0)
        back_idx = torch.cat([p.idx for p in back_packets], dim=0)
    else:
        back_data = torch.zeros((0, H), device=dev())
        back_idx = torch.zeros((0,), device=dev(), dtype=torch.long)

    # Filter only indices that belong to my range
    mask_mine = (back_idx >= start) & (back_idx < end)
    mine_data = back_data[mask_mine]
    mine_idx = back_idx[mask_mine] - start

    # Place into local output buffer (my shard of the flat output)
    local_out = torch.zeros((myN, H), device=dev())
    if mine_data.numel() > 0:
        local_out[mine_idx] = mine_data

    print0(f"[MoE] rank={rank()} local_out shape={tuple(local_out.shape)} | tokens shard={myN}")

    cleanup_dist()


# -----------------------------
# Entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, required=True,
                        choices=["data", "tensor", "pipeline", "sequence", "moe"],
                        help="Which example to run")
    parser.add_argument("--micro_batches", type=int, default=4, help="Pipeline micro-batches")
    args = parser.parse_args()

    ex = args.example
    if ex == "data":
        run_data_batching()
    elif ex == "tensor":
        run_tensor_parallel()
    elif ex == "pipeline":
        run_pipeline(micro_batches=args.micro_batches)
    elif ex == "sequence":
        run_sequence_parallel()
    elif ex == "moe":
        run_moe_parallel()
    else:
        print("Unknown example", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
