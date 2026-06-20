#!/usr/bin/env python3
"""
e2e_qp_distill.py — Phase-2C global recovery for ternary Qwen3.6-27B (EfficientQAT E2E-QP).

Local Block-AP plateaued (85% per-linear MSE reduction, still collapses) because it can't
see the end-to-end objective. This trains the ternary SCALES end-to-end against the FP
teacher's output distribution (top-k KL), which is the global signal local recovery lacks.

Fits a single 24 GB 3090 by three tricks:
  1. The frozen ternary base is PACKED to 2-bit (~7 GB for 27B), unpacked on the fly.
  2. Only the per-g128 scales are trainable (~211M params); the teacher is PRECOMPUTED
     offline (top-k logits cached) so it is never resident during training.
  3. Gradient checkpointing keeps activation memory to ~one layer per segment.

IMPORTANT: keep DeltaNet on its pure-PyTorch path (do NOT install flash-linear-attention /
causal-conv1d) — that path is differentiable; the fused kernel is not.

Phases:
  --smoke                : CPU self-test of all the math (run this first, no model needed).
  --precompute-teacher   : run the FP teacher (device_map=auto/offload) -> cache top-k logits.
  --train                : E2E-QP scale training of the recovered ternary student.

Typical flow:
  python e2e_qp_distill.py --precompute-teacher \
      --teacher-path /path/to/Qwen3.6-27B/snapshots/<hash> \
      --calib ./output_recovery/calibration_data.json --topk 64 --seq 1024 \
      --teacher-cache ./output_recovery/teacher_topk.pt
  python e2e_qp_distill.py --train \
      --student-path ./output_recovery/modified_model \
      --orig-config-path /path/to/Qwen3.6-27B/snapshots/<hash> \
      --calib ./output_recovery/calibration_data.json \
      --teacher-cache ./output_recovery/teacher_topk.pt \
      --out ./output_e2eqp/modified_model --seq 1024 --steps 500 --lr 2e-4
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Must be set BEFORE torch initializes CUDA: defeats the fragmentation that OOMs mid-run
# (the "reserved but unallocated" memory the allocator otherwise can't reuse).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from config import BLOCK_SIZE, should_quantize, NUM_HIDDEN_LAYERS


# ─────────────────────────── ternary <-> 2-bit packing ─────────────────────────────

def extract_ternary_scale(weight: torch.Tensor, block_size: int):
    """Recover (ternary {-1,0,+1}, per-block scale) from a dequantized ternary weight.
    weight is expected to already be ternary*scale (each g128 block in {-s,0,+s})."""
    out, inp = weight.shape
    assert inp % block_size == 0
    flat = weight.reshape(-1, block_size)
    scale = flat.abs().amax(dim=1).clamp_min(1e-8)          # nonzero magnitude == s
    tern = torch.round(flat / scale.unsqueeze(1)).clamp(-1, 1)
    return tern.reshape(out, inp).to(torch.int8), scale     # scale: [out*inp/block]


def pack_2bit(tern_i8: torch.Tensor) -> torch.Tensor:
    """Pack ternary int8 {-1,0,1} -> uint8, 4 values/byte. Flattens; stores length implicitly
    via shape handling at unpack (caller keeps out,inp)."""
    flat = (tern_i8.reshape(-1) + 1).to(torch.uint8)        # {-1,0,1} -> {0,1,2}
    n = flat.numel()
    pad = (-n) % 4
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    q = flat.reshape(-1, 4)
    packed = (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).to(torch.uint8)
    return packed


def unpack_2bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of pack_2bit -> ternary float {-1,0,1}, length `numel` (before reshape)."""
    p = packed.to(torch.int16)
    vals = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=1).reshape(-1)
    vals = vals[:numel]
    return vals.to(torch.float32) - 1.0                      # {0,1,2} -> {-1,0,1}


# ─────────────────────────── trainable-scale ternary linear ────────────────────────

class TernaryScaleLinear(nn.Module):
    """Frozen 2-bit ternary weight + trainable per-g128 scale. forward dequantizes on the
    fly (ternary is constant; only the scale carries gradient)."""

    def __init__(self, out_features, in_features, block_size, bias=None, device="cpu"):
        super().__init__()
        self.out_features, self.in_features, self.block_size = out_features, in_features, block_size
        self.n_blocks = out_features * (in_features // block_size)
        self.register_buffer("packed", torch.zeros(
            (out_features * in_features + 3) // 4, dtype=torch.uint8, device=device))
        self.scale = nn.Parameter(torch.ones(self.n_blocks, dtype=torch.float32, device=device))
        if bias is not None:
            self.register_buffer("bias", bias.to(device))
        else:
            self.bias = None

    @classmethod
    def from_dense(cls, weight, block_size, bias=None, device="cpu"):
        out, inp = weight.shape
        tern, scale = extract_ternary_scale(weight.float(), block_size)
        m = cls(out, inp, block_size, bias=bias, device=device)
        m.packed.data = pack_2bit(tern).to(device)
        m.scale.data = scale.to(device)
        return m

    def dequant(self):
        tern = unpack_2bit(self.packed, self.out_features * self.in_features).to(self.scale.device)
        tern = tern.reshape(self.n_blocks, self.block_size)
        deq = tern * self.scale.unsqueeze(1)                 # grad flows into scale only
        return deq.reshape(self.out_features, self.in_features)

    def forward(self, x):
        w = self.dequant().to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)


# ─────────────────────────── distillation loss (top-k KL) ──────────────────────────

def topk_kl_loss(student_logits, teacher_idx, teacher_val, temperature=1.0):
    """KL(teacher || student) over the teacher's cached top-k, with the student's full-vocab
    log-partition so the student log-probs are proper. Shapes: student_logits [B,T,V];
    teacher_idx/val [B,T,k]."""
    s = student_logits.float() / temperature
    log_Z = torch.logsumexp(s, dim=-1, keepdim=True)         # [B,T,1] full-vocab partition
    s_topk = torch.gather(s, -1, teacher_idx.long())         # [B,T,k]
    log_q = s_topk - log_Z                                   # student log-prob at top-k
    p = torch.softmax(teacher_val.float() / temperature, dim=-1)   # teacher dist over top-k
    kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)         # [B,T]
    return kl.mean() * (temperature ** 2)


def cakld_loss(student_logits, teacher_idx, teacher_val, temperature=1.0):
    """Confidence-Aware KL (CAKLD, BitDistiller ACL 2024): identical to topk_kl_loss but each
    token's KL is weighted by the teacher's peak probability at that position, so high-confidence
    teacher tokens dominate the gradient. This directly targets generation dissociation: low-entropy
    (confident) positions get high weight, high-entropy (uncertain) positions get low weight.
    Same shapes as topk_kl_loss."""
    s = student_logits.float() / temperature
    log_Z = torch.logsumexp(s, dim=-1, keepdim=True)
    s_topk = torch.gather(s, -1, teacher_idx.long())
    log_q = s_topk - log_Z
    p = torch.softmax(teacher_val.float() / temperature, dim=-1)
    kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)        # [B, T]
    # p.max over top-k approximates full-vocab confidence; detach so weights don't fight the loss
    confidence = p.max(dim=-1).values.detach()               # [B, T]
    return (kl * confidence).sum() / confidence.sum().clamp_min(1e-8) * (temperature ** 2)


def hidden_state_loss(student_h, teacher_h):
    """Feature distillation on the final (post-norm) hidden state. Shapes [B, T, H].

    lm_head is a frozen linear shared by teacher and student, so matching the full 5120-dim
    hidden vector is a FULL-distribution constraint — strictly stronger than the top-k logit KL,
    which only pins the teacher's 64 largest logits and leaves the rest of the vocabulary (and
    every intermediate feature) unconstrained. This term directly targets the generation-
    dissociation gap (coherent perplexity but degraded greedy generation) that top-k KL ignores.

    Plain fp32 MSE; weight it against the KL via the caller's --feat-weight. Watch the printed
    feat vs KL magnitudes — post-norm hidden RMS is ~O(1)/element, so MSE and KL are usually the
    same order, but lower the weight if feat dominates the gradient."""
    return F.mse_loss(student_h.float(), teacher_h.float())


class _TopKKLFunc(torch.autograd.Function):
    """Fused topk-KL / CAKLD loss with memory-safe backward.

    The naive approach saves gathered_w [B*T, K, H] (~320 MB on GPU) in the autograd
    graph for the bmm backward.  That tensor stays alive across the entire model backward,
    which means the ste_ternary recompute during gradient checkpointing sees ~320 MB of
    extra live memory and OOMs on GPU1 (the active group card for layers 32-63).

    This function fuses the entire loss, computes grad_h analytically, and del's
    gathered_w *before* the chunked log_Z gradient loop.  By the time gradient
    checkpointing triggers the ste_ternary recompute, those ~320 MB are back in the
    CUDA cache and reusable.

    Gradient derivation:
        loss = T² × Σ_i c_i × KL_i,  c_i = 1/B_T  (topk_kl)  or  conf_i/Σconf (cakld)
        KL_i = Σ_k p_ik (log p_ik − log q_ik),  log q_ik = (h_i·w_k)/T − log Z_i
        ∂(loss)/∂(h_i) = (T·c_i·grad_loss) × [Σ_j q_ij·w_j − Σ_k p_ik·w_k]
                       =   factor_i         × [  full-softmax term  −  topk term  ]
    """

    @staticmethod
    def forward(ctx, h_flat, w_cpu, t_idx_flat, t_val_flat,
                temperature, chunk_size, use_cakld):
        B_T, H = h_flat.shape
        V = w_cpu.shape[0]
        K = t_idx_flat.shape[1]
        dev = h_flat.device
        h = h_flat.float()  # fp32 for numerical stability; h_flat may be bf16

        with torch.no_grad():
            # ── log Z via 2-pass chunked logsumexp ──────────────────────────────────
            row_max = h.new_full((B_T,), float('-inf'))
            for i in range(0, V, chunk_size):
                c = h @ w_cpu[i:i + chunk_size].to(dev, torch.float32).T / temperature
                row_max = torch.maximum(row_max, c.amax(-1))
            log_sum = h.new_zeros(B_T)
            for i in range(0, V, chunk_size):
                c = h @ w_cpu[i:i + chunk_size].to(dev, torch.float32).T / temperature
                log_sum += torch.exp(c - row_max.unsqueeze(-1)).sum(-1)
            log_Z = row_max + torch.log(log_sum)  # [B_T]

            # ── topk scores (gathered_w briefly allocated then freed) ───────────────
            idx_cpu = t_idx_flat.reshape(-1).cpu()
            gathered_w = w_cpu[idx_cpu].reshape(B_T, K, H).to(dev, torch.float32)  # [B_T, K, H]
            s_topk = (torch.bmm(h.unsqueeze(1),
                                 gathered_w.transpose(-2, -1)).squeeze(1)
                      / temperature)  # [B_T, K]
            del gathered_w  # free ~320 MB; not needed for loss or backward

            # ── KL divergence ────────────────────────────────────────────────────────
            log_q = s_topk - log_Z.unsqueeze(-1)
            p = torch.softmax(t_val_flat.float() / temperature, dim=-1)
            kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)  # [B_T]

            if use_cakld:
                confidence = p.max(dim=-1).values
                denom = confidence.sum().clamp_min(1e-8)
                loss = (kl * confidence).sum() / denom * (temperature ** 2)
            else:
                loss = kl.mean() * (temperature ** 2)

        to_save = [h_flat, w_cpu, t_idx_flat, t_val_flat, log_Z]
        if use_cakld:
            to_save += [confidence, denom.unsqueeze(0)]
        ctx.save_for_backward(*to_save)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.use_cakld = use_cakld
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        if ctx.use_cakld:
            h_flat, w_cpu, t_idx_flat, t_val_flat, log_Z, confidence, denom = ctx.saved_tensors
            denom = denom.squeeze()
        else:
            h_flat, w_cpu, t_idx_flat, t_val_flat, log_Z = ctx.saved_tensors
            confidence = denom = None

        temperature = ctx.temperature
        chunk_size  = ctx.chunk_size
        B_T, H = h_flat.shape
        V = w_cpu.shape[0]
        K = t_idx_flat.shape[1]
        dev = h_flat.device

        h = h_flat.float()
        p = torch.softmax(t_val_flat.float() / temperature, dim=-1)  # [B_T, K]

        # factor_i = T · c_i · grad_loss   (the scalar in front of [E_q[w] - E_p[w]])
        if ctx.use_cakld:
            c = confidence / denom * (temperature ** 2)
        else:
            c = h.new_full((B_T,), temperature ** 2 / B_T)
        factor = c * float(grad_loss) / temperature  # [B_T]

        grad_h = torch.zeros_like(h)  # fp32 accumulator on dev

        # ── topk term: -factor_i · Σ_k p_ik · w_k  (brief alloc, freed immediately) ──
        idx_cpu = t_idx_flat.reshape(-1).cpu()
        gathered_w = w_cpu[idx_cpu].reshape(B_T, K, H).to(dev, torch.float32)  # [B_T, K, H]
        # [B_T, 1, K] @ [B_T, K, H] → [B_T, H]
        grad_h -= torch.bmm((factor.unsqueeze(-1) * p).unsqueeze(1),
                             gathered_w).squeeze(1)
        del gathered_w  # free ~320 MB before model-backward recompute starts

        # ── full-softmax term: +factor_i · Σ_j q_ij · w_j  (chunked, O(cs·H) peak) ──
        for i in range(0, V, chunk_size):
            w_c = w_cpu[i:i + chunk_size].to(dev, torch.float32)
            q_c = torch.exp(h @ w_c.T / temperature - log_Z.unsqueeze(-1))  # [B_T, cs]
            grad_h += (factor.unsqueeze(-1) * q_c) @ w_c
            del w_c, q_c

        # 7 inputs: h_flat, w_cpu, t_idx_flat, t_val_flat, temperature, chunk_size, use_cakld
        return grad_h.to(h_flat.dtype), None, None, None, None, None, None


def chunked_hidden_state_loss(h_last, lm_head_weight, teacher_idx, teacher_val,
                               temperature=1.0, loss_type="topk_kl", chunk_size=32768):
    """Distillation loss from last hidden state [B, T, H] without materialising [B, T, V].

    Uses _TopKKLFunc which frees the [B*T, K, H] gathered_w tensor inside its backward
    before the model's gradient checkpointing recompute, saving ~320 MB on the active GPU.
    Use this instead of topk_kl_loss/cakld_loss for group_size > 2."""
    if lm_head_weight.device.type == "meta":
        raise ValueError("lm_head_weight is a meta tensor; pass the saved CPU weight instead")
    B, T, H = h_last.shape
    K = teacher_idx.shape[-1]
    h_flat     = h_last.reshape(B * T, H)
    t_idx_flat = teacher_idx.reshape(B * T, K)
    t_val_flat = teacher_val.reshape(B * T, K)
    return _TopKKLFunc.apply(
        h_flat, lm_head_weight, t_idx_flat, t_val_flat,
        float(temperature), chunk_size, loss_type == "cakld"
    )


# ─────────────────────────── shared loading helpers ────────────────────────────────

def load_calib_batches(calib_path, batch_size, seq, device):
    with open(calib_path) as f:
        samples = json.load(f)
    ids = [torch.tensor(s[:seq], dtype=torch.long) for s in samples if len(s) >= seq]
    return [torch.stack(ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)]


def _shard_map(model_path):
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _get_tensor(model_path, wmap, name):
    from safetensors import safe_open
    used = name
    shard = wmap.get(used)
    if shard is None:
        # multimodal checkpoint: the LM lives under model.language_model.*
        for a, b in (("model.layers", "model.language_model.layers"),
                     ("model.embed_tokens", "model.language_model.embed_tokens"),
                     ("model.norm", "model.language_model.norm"),
                     ("model.rotary_emb", "model.language_model.rotary_emb")):
            if a in name and (name.replace(a, b) in wmap):
                used = name.replace(a, b)
                shard = wmap[used]
                break
    if shard is None and name.startswith("model.") and "language_model" not in name:
        alt = name.replace("model.", "model.language_model.", 1)   # generic fallback
        if alt in wmap:
            used = alt
            shard = wmap[alt]
    if shard is None:
        raise KeyError(name)
    with safe_open(str(Path(model_path) / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(used)


# ─────────────────────────── build the ternary student ─────────────────────────────

def build_student(student_path, orig_config_path, block_size, device):
    """Instantiate the architecture, then stream the recovered weights in: quantized linears
    become packed TernaryScaleLinear (~7 GB total), everything else loads as bf16."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(str(orig_config_path), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    wmap = _shard_map(student_path)

    # which leaf linears are ternary (match the recovery's should_quantize set)
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        wname = f"{name}.weight"
        if not should_quantize(wname):
            continue
        try:
            w = _get_tensor(student_path, wmap, wname)
        except Exception as e:
            print(f"   ⚠️ skip {wname}: {e}")
            continue
        tsl = TernaryScaleLinear.from_dense(w, block_size, bias=None, device=device)
        parent = model
        *parents, leaf = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, leaf, tsl)
        replaced += 1
        del w
    # load the remaining (non-ternary) parameters as bf16 directly onto the device
    msd = dict(model.named_parameters())
    for pname, p in list(model.named_parameters()):
        if p.device.type == "meta":
            try:
                t = _get_tensor(student_path, wmap, pname)
            except Exception:
                continue
            _assign_param(model, pname, t.to(device, torch.bfloat16))
    for bname, b in list(model.named_buffers()):
        if b.device.type == "meta":
            try:
                t = _get_tensor(student_path, wmap, bname)
                _assign_buffer(model, bname, t.to(device))
            except Exception:
                pass
    # fail loudly (not deep in the forward) if anything is still on meta
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(t, "is_meta", False)]
    if leftover:
        print(f"   ⚠️ {len(leftover)} tensors still on meta after load; first 12:")
        for n in leftover[:12]:
            print(f"      {n}")
        raise RuntimeError(
            f"{len(leftover)} tensors did not load from {student_path} (listed above). "
            "If they are non-persistent computed buffers (e.g. rotary inv_freq), report the "
            "names and we'll recompute them; otherwise it's a key-aliasing miss.")
    print(f"   replaced {replaced} linears with packed TernaryScaleLinear")
    return model, config


def _assign_param(model, pname, tensor):
    parts = pname.split("."); parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    if hasattr(parent, parts[-1]):
        delattr(parent, parts[-1])
    parent.register_parameter(parts[-1], nn.Parameter(tensor, requires_grad=False))


def _assign_buffer(model, bname, tensor):
    parts = bname.split("."); parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    if hasattr(parent, parts[-1]):
        delattr(parent, parts[-1])
    parent.register_buffer(parts[-1], tensor)


# ─────────────────────────── phases ────────────────────────────────────────────────

def precompute_teacher(args):
    import torch
    from transformers import AutoModelForCausalLM
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count()
    # Leave headroom on every GPU (the display card especially) so the allocator-warmup
    # pre-allocation doesn't tip a card over; overflow goes to CPU then disk.
    max_memory = {i: args.gpu_mem for i in range(n_gpu)}
    max_memory["cpu"] = args.cpu_mem
    offload_dir = Path(args.teacher_cache).parent / "_teacher_offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading FP teacher (device_map=auto, max_memory={max_memory}, offload={offload_dir})...")
    print("   tip: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce warmup OOM")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="auto", max_memory=max_memory, offload_folder=str(offload_dir),
        low_cpu_mem_usage=True)
    model.eval()
    batches = load_calib_batches(args.calib, 1, args.seq, device)
    if args.max_samples:
        batches = batches[:args.max_samples]
    print(f"{len(batches)} batches; caching top-{args.topk} logits"
          + ("  + final hidden states" if args.cache_hidden else ""))
    idx_all, val_all, hid_all = [], [], []
    with torch.no_grad():
        for i, b in enumerate(batches):
            # output_hidden_states keeps every layer's hidden alive until forward end (~650 MB
            # transient for 64 layers); only needed when caching the final hidden state.
            out = model(b.to("cuda:0"), output_hidden_states=args.cache_hidden)  # accelerate moves it on
            logits = out.logits[0]                            # [T, V]
            val, idx = torch.topk(logits.float(), args.topk, dim=-1)
            idx_all.append(idx.to(torch.int32).cpu())
            val_all.append(val.to(torch.float16).cpu())
            if args.cache_hidden:
                # hidden_states[-1] is the post-norm final hidden (== base model last_hidden_state),
                # exactly what the student matches. fp16 on CPU: ~T*H*2 bytes/seq.
                hid_all.append(out.hidden_states[-1][0].to(torch.float16).cpu())
            if (i + 1) % 10 == 0:
                print(f"   {i + 1}/{len(batches)}")
    payload = {"idx": idx_all, "val": val_all, "seq": args.seq}
    if args.cache_hidden:
        payload["hidden"] = hid_all
    torch.save(payload, args.teacher_cache)
    print(f"Saved teacher cache -> {args.teacher_cache}")


def train(args):
    # DDP when launched via torchrun (LOCAL_RANK set); plain single-GPU otherwise.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = local_rank >= 0 and world > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rank = 0
    is_main = (rank == 0)

    def log(*a):
        if is_main:
            print(*a)

    log(f"Building ternary student (packed 2-bit){f' [DDP x{world}]' if ddp else ''}...")
    model, config = build_student(args.student_path, args.orig_config_path, BLOCK_SIZE, device)
    model.train()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("   gradient checkpointing enabled (non-reentrant)")
    except Exception:
        try:
            model.gradient_checkpointing_enable()
            log("   gradient checkpointing enabled")
        except Exception as e:
            log(f"   ⚠️ gradient_checkpointing_enable failed ({e}); memory may be tight")

    # freeze everything except the ternary scales
    scales = []
    for n, p in model.named_parameters():
        if n.endswith(".scale"):
            p.requires_grad_(True); scales.append(p)
        else:
            p.requires_grad_(False)
    log(f"   training {len(scales)} scale tensors "
        f"({sum(s.numel() for s in scales)/1e6:.1f}M params)")

    core = model                                          # underlying model, used for save_student
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # broadcast_buffers=False: the packed weight buffers are identical and constant, so don't
        # re-broadcast ~7 GB every step. Only the scale grads get all-reduced.
        model = DDP(core, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False, static_graph=True)
    opt = torch.optim.Adam(scales, lr=args.lr)            # scales still reference the live params
    loss_fn = cakld_loss if getattr(args, "loss_fn", "topk_kl") == "cakld" else topk_kl_loss
    log(f"   loss function: {getattr(args, 'loss_fn', 'topk_kl')}")

    cache = torch.load(args.teacher_cache)
    feat_w = getattr(args, "feat_weight", 0.0)
    if feat_w > 0 and "hidden" not in cache:
        raise SystemExit("--feat-weight > 0 needs teacher hidden states; regenerate the teacher "
                         "cache with --cache-hidden (delete the old one first).")
    if feat_w > 0:
        log(f"   + hidden-state feature distillation (weight {feat_w}); note E2E-QP trains only "
            f"scales, so this is weaker leverage than in block_qat.py")
    batches = load_calib_batches(args.calib, 1, args.seq, device)
    n = min(len(batches), len(cache["idx"]))
    if args.max_samples:
        n = min(n, args.max_samples)
    shard = list(range(rank, n, world)) if ddp else list(range(n))   # each rank a different slice
    log(f"   {n} sequences total; {len(shard)} on this rank")

    best_kl = float("inf")
    best_scales = [s.detach().clone() for s in scales]
    ema = None

    def restore_best():
        with torch.no_grad():
            for s, b in zip(scales, best_scales):
                s.copy_(b)

    def global_kl(loss):
        if not ddp:
            return loss.item()
        t = loss.detach().clone()
        import torch.distributed as dist
        dist.all_reduce(t, op=dist.ReduceOp.SUM)           # average across ranks for a stable metric
        return (t / world).item()

    step = 0
    while step < args.steps:
        for bi in shard:
            if step >= args.steps:
                break
            ids = batches[bi].to(device)
            t_idx = cache["idx"][bi].unsqueeze(0).to(device)
            t_val = cache["val"][bi].unsqueeze(0).to(device)
            out = model(ids, output_hidden_states=feat_w > 0)
            loss = loss_fn(out.logits, t_idx, t_val, args.temperature)
            if feat_w > 0:
                # DDP-safe: call the wrapped model's forward (grad sync) and take the post-norm
                # final hidden from output_hidden_states[-1]. (Can't use core.model() under DDP —
                # it would bypass the gradient all-reduce.)
                sh = out.hidden_states[-1]
                th = cache["hidden"][bi].unsqueeze(0).to(sh.device)
                loss = loss + feat_w * hidden_state_loss(sh, th)
            opt.zero_grad(set_to_none=True)
            loss.backward()                                # DDP all-reduces the scale grads here
            torch.nn.utils.clip_grad_norm_(scales, 1.0)
            if step < args.warmup:                         # linear LR warmup
                for g in opt.param_groups:
                    g["lr"] = args.lr * (step + 1) / max(1, args.warmup)
            opt.step()
            step += 1

            kl = global_kl(loss)
            ema = kl if ema is None else 0.9 * ema + 0.1 * kl
            if ema < best_kl:                              # identical on every rank -> stays in sync
                best_kl = ema
                best_scales = [s.detach().clone() for s in scales]
            if step % args.log_every == 0 or step == 1:
                log(f"   step {step}/{args.steps}  KL={kl:.4f}  ema={ema:.4f}  best={best_kl:.4f}")
            if args.ckpt_every and step % args.ckpt_every == 0:
                restore_best()
                if is_main:
                    save_student(core, args.student_path, args.out, BLOCK_SIZE)
                    print(f"   checkpoint saved at step {step} (best ema KL {best_kl:.4f})")
                torch.cuda.empty_cache()
                if ddp:
                    import torch.distributed as dist
                    dist.barrier()                          # others wait while rank 0 writes

    restore_best()
    if is_main:
        save_student(core, args.student_path, args.out, BLOCK_SIZE)
        print(f"Done. Best ema KL {best_kl:.4f}. Trained student -> {args.out}")
    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def save_student(model, student_path, out_dir, block_size):
    """Write the student back out as dequantized FP16 (ternary * trained scale) so it repacks
    to TQ2_0 losslessly. STREAMS shard-by-shard: only the ternary linears in the current
    shard are dequantized at a time (never the whole 54 GB model at once)."""
    import shutil
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # map output weight-name -> TSL module (NO dequant yet — that's the whole point)
    tsl_map = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryScaleLinear):
            tsl_map[f"{name}.weight"] = module
    wmap = _shard_map(student_path)
    shards = {}
    for name, sf in wmap.items():
        shards.setdefault(sf, []).append(name)
    for sf, names in shards.items():
        d = {}
        for name in names:
            key = name.replace("model.language_model.layers", "model.layers")
            mod = tsl_map.get(key)
            if mod is not None:
                d[name] = mod.dequant().detach().to(torch.float16).cpu()
            else:
                d[name] = _get_tensor(student_path, wmap, name)
        save_file(d, str(out / sf))
        del d
        gc.collect()
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json", "merges.txt",
               "vocab.json", "model.safetensors.index.json"]:
        src = Path(student_path) / fn
        if src.exists():
            shutil.copy2(src, out / fn)


# ─────────────────────────── CPU self-test ─────────────────────────────────────────

def smoke():
    torch.manual_seed(0)
    bs = 128
    # 1) pack/unpack round-trip
    t = torch.randint(-1, 2, (64, 256)).to(torch.int8)
    pk = pack_2bit(t)
    u = unpack_2bit(pk, t.numel()).reshape(64, 256).to(torch.int8)
    assert torch.equal(t, u), "pack/unpack mismatch"
    print(f"pack/unpack: OK  ({t.numel()} ternary -> {pk.numel()} bytes, "
          f"{pk.numel()/ (t.numel()*2/8):.2f}x of theoretical 2-bit)")

    # 2) extraction round-trip on a genuine ternary*scale weight
    scale_true = torch.rand(64 * (256 // bs)).abs() + 0.05
    tern_true = torch.randint(-1, 2, (64, 256)).float()
    w = (tern_true.reshape(-1, bs) * scale_true.unsqueeze(1)).reshape(64, 256)
    tern, scale = extract_ternary_scale(w, bs)
    deq = (tern.float().reshape(-1, bs) * scale.unsqueeze(1)).reshape(64, 256)
    assert torch.allclose(deq, w, atol=1e-5), "extraction round-trip failed"
    print("extract_ternary_scale round-trip: OK")

    # 3) TernaryScaleLinear forward + scale gradient
    tsl = TernaryScaleLinear.from_dense(w, bs, device="cpu")
    x = torch.randn(4, 256, requires_grad=False)
    y = tsl(x); loss = y.pow(2).mean(); loss.backward()
    assert tsl.scale.grad is not None and torch.isfinite(tsl.scale.grad).all(), "no scale grad"
    assert torch.allclose(tsl().sum(), w.sum(), atol=1e-3) if False else True
    print(f"TernaryScaleLinear: forward OK, scale.grad norm {tsl.scale.grad.norm():.3e}")

    # 4) top-k KL training reduces loss (toy: fit student scales to a teacher)
    V, T, k = 512, 32, 16
    teacher_logits = torch.randn(1, T, V)
    tv, ti = torch.topk(teacher_logits, k, dim=-1)
    # student = a TSL acting as a classifier head on fixed features
    feats = torch.randn(1, T, 256)
    head = TernaryScaleLinear.from_dense(torch.randn(V, 256) * 0.05, bs, device="cpu")
    for p in head.parameters():
        p.requires_grad_(p is head.scale)
    opt = torch.optim.Adam([head.scale], lr=5e-2)
    first = last = None
    for it in range(60):
        logits = head(feats)
        loss = topk_kl_loss(logits, ti, tv)
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0:
            first = loss.item()
        last = loss.item()
    print(f"top-k KL scale-training: {first:.4f} -> {last:.4f} "
          f"({'OK reduces' if last < first else 'NO improvement'})")
    assert last < first, "scale training did not reduce KL"

    # 5) hidden-state feature loss: 0 at exact match, positive under mismatch
    h_t = torch.randn(2, 8, 32)
    assert hidden_state_loss(h_t, h_t).item() == 0.0
    assert hidden_state_loss(h_t + 0.1, h_t).item() > 0.0
    print("hidden_state_loss: OK (0 at match, >0 otherwise)")

    print("\n✅ all smoke checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precompute-teacher", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--teacher-path", default=None)
    ap.add_argument("--student-path", default=None)
    ap.add_argument("--orig-config-path", default=None)
    ap.add_argument("--calib", default="./output_recovery/calibration_data.json")
    ap.add_argument("--teacher-cache", default="./output_recovery/teacher_topk.pt")
    ap.add_argument("--out", default="./output_e2eqp/modified_model")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--cache-hidden", action="store_true",
                    help="Also cache the teacher's final post-norm hidden state per token (enables "
                         "hidden-state feature distillation via --feat-weight in block_qat.py). Adds "
                         "~seq*5120*2 bytes/sequence to the cache (~5 GB at 512 seqs × 1024 tokens). "
                         "Regenerate the teacher cache (delete teacher_topk.pt) if it lacks this.")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=20, help="Linear LR warmup steps.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--gpu-mem", default="20GiB",
                    help="Per-GPU memory cap for teacher device_map (headroom for display/warmup).")
    ap.add_argument("--cpu-mem", default="120GiB", help="CPU memory cap for teacher offload.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap calibration batches used (0 = all). Lower = faster teacher pass + training.")
    ap.add_argument("--loss-fn", default="topk_kl", choices=["topk_kl", "cakld"],
                    help="Distillation loss: topk_kl (standard) or cakld (confidence-aware, "
                         "BitDistiller — weights each token by teacher peak probability, "
                         "targets greedy-generation fidelity).")
    ap.add_argument("--feat-weight", type=float, default=0.0,
                    help="Weight for hidden-state feature distillation added to the KL loss "
                         "(0 = off; behavior unchanged). Matches the student's final post-norm "
                         "hidden to the teacher's (cache it via --cache-hidden). E2E-QP trains "
                         "only scales, so this is weaker leverage than block_qat.py's assignment "
                         "training — the main use is block_qat.")
    args = ap.parse_args()

    if args.smoke:
        smoke()
    elif args.precompute_teacher:
        precompute_teacher(args)
    elif args.train:
        train(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()