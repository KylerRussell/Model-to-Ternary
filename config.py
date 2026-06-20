"""
Configuration for Qwen3.6-27B → Ternary conversion pipeline.

This module defines the model structure, layer targeting rules, and
quantization parameters for converting Qwen3.6-27B (architecture:
Qwen3_5ForConditionalGeneration) to a 1.58-bit ternary representation.

Architecture notes:
  - 64 layers in a 3:1 hybrid pattern (48 Gated DeltaNet + 16 Full Attention)
  - hidden_size=5120, intermediate_size=17408
  - vocab_size=248320, tie_word_embeddings=False
  - Includes vision encoder (27-depth ViT) - preserved in FP16
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Model identity
# ──────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_ARCH = "Qwen3_5ForConditionalGeneration"

# ──────────────────────────────────────────────────────────────────────
# Layer pattern (from config.json layer_types)
# ──────────────────────────────────────────────────────────────────────
NUM_HIDDEN_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4  # every 4th layer is full_attention

LAYER_TYPES = []
for i in range(NUM_HIDDEN_LAYERS):
    if (i + 1) % FULL_ATTENTION_INTERVAL == 0:
        LAYER_TYPES.append("full_attention")
    else:
        LAYER_TYPES.append("linear_attention")

# ──────────────────────────────────────────────────────────────────────
# Dimension constants
# ──────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
VOCAB_SIZE = 248320

# Full attention params
NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256

# DeltaNet (linear attention) params
LINEAR_KEY_HEAD_DIM = 128
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 48
LINEAR_VALUE_HEAD_DIM = 128

# ──────────────────────────────────────────────────────────────────────
# Quantization parameters
# ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 256      # per-block scale granularity. MUST match the TQ2_0 deploy grid (one fp16
                      # scale per 256 weights) so training and export use the SAME grid — training
                      # at 128 then exporting at 256 collapsed every scale-pair (+24pt ppl ratio,
                      # measured). 256 divides every quantized in_features. Block-AP, E2E-QP, and
                      # QAT all read this, so the whole pipeline is now g256-native end to end.
TERNARY_VALUES = (-1, 0, 1)

# ──────────────────────────────────────────────────────────────────────
# Layer targeting rules
# ──────────────────────────────────────────────────────────────────────

# Tensor name patterns to QUANTIZE (applied to both layer types)
# These are the nn.Linear projections in attention and MLP
QUANTIZE_PATTERNS = [
    # Attention projections (both full_attention and linear_attention)
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    # DeltaNet-specific projections (linear attention)
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_qkvz.weight",
    ".linear_attn.out_proj.weight",
    # MLP projections
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
]

# Tensor name patterns to KEEP in FP16 (never quantize)
KEEP_FP16_PATTERNS = [
    "embed_tokens",       # token embeddings
    "lm_head",            # language model head
    "visual",             # entire vision encoder
    "merger",             # vision-language merger
    "norm.weight",        # all RMSNorm/LayerNorm weights
    "layernorm",          # any layernorm variants
    "rotary_emb",         # rotary embedding tables
    "mtp_",               # multi-token prediction head
    "conv1d",             # DeltaNet conv1d kernels (small, keep precise)
    ".bias",              # any biases (usually small)
]

# ── DeltaNet isolation experiment (DIAGNOSTIC) ──────────────────────────────
# (Experiment 1 — collapsed; set False now.) When True, all 48 DeltaNet attention
# projections (linear_attn.*) stay FP16 (rotated, not ternarized).
ISOLATE_DELTANET_FP16 = False
if ISOLATE_DELTANET_FP16:
    KEEP_FP16_PATTERNS = KEEP_FP16_PATTERNS + [".linear_attn"]
# ────────────────────────────────────────────────────────────────────────────

# ── Experiment 2: protect UN-ROTATED-input projections (DIAGNOSTIC) ─────────
# (Experiment 2 — collapsed; set False now.) When True (with experiment 1), only the
# rotated-input projections {q,k,v,gate,up} are ternarized.
PROTECT_UNROTATED_INPUTS = False
if PROTECT_UNROTATED_INPUTS:
    KEEP_FP16_PATTERNS = KEEP_FP16_PATTERNS + [".o_proj", ".down_proj"]
# ────────────────────────────────────────────────────────────────────────────

# ── Mixed precision: keep full-attention layers at FP16 ─────────────────────
# When True, the 16 full-attention layers' self_attn projections (q/k/v/o) stay FP16;
# only the 48 DeltaNet layers and all MLP projections are ternarized.
# Rationale (Quamba/Q-Mamba literature): linear-attention/SSM layers carry larger output
# outliers and are more quantization-sensitive than standard attention — ternarizing the
# DeltaNet layers while keeping full-attention at FP16/Q4 gives disproportionate quality
# for a small bpw increase, and is fully compatible with GGUF TQ2_0 as a per-tensor choice.
# NOTE: this changes the quantization target — a new Phase 1→3 run is required.
MIXED_PRECISION_FULL_ATTN = False
# ────────────────────────────────────────────────────────────────────────────

# ── Experiment 3: bug-vs-ceiling — quantize ONLY a few layers (DIAGNOSTIC) ──
# Quantize the normal target tensors ONLY in these layer indices; every other layer
# stays FP16-rotated. Ternarizing a few standard (full-attention) layers out of 64
# should cause only minor degradation in a HEALTHY model. If even this small set
# collapses generation into garbage, a handful of tensors is destroying the whole
# model — a bug in the quantization/save path, not a quality ceiling. If it stays
# coherent, widen the set ({3,7,11} → first 8 → first 16 → …) to map the compounding
# cliff. Set to None to quantize all layers.
QUANTIZE_ONLY_LAYERS = None   # None = all layers (production). {3,7,11} etc. = diagnostic only.


def _layer_index(name: str):
    """Extract integer layer index from '...layers.<i>...', else None."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None
# ────────────────────────────────────────────────────────────────────────────


def should_quantize(tensor_name: str) -> bool:
    """Determine if a tensor should be quantized to ternary."""
    # First check exclusions (higher priority)
    for pattern in KEEP_FP16_PATTERNS:
        if pattern in tensor_name:
            return False
    # Then check if it matches a quantization target
    for pattern in QUANTIZE_PATTERNS:
        if pattern in tensor_name:
            # Experiment-3 gate: restrict quantization to specific layers if set
            if QUANTIZE_ONLY_LAYERS is not None:
                li = _layer_index(tensor_name)
                if li is None or li not in QUANTIZE_ONLY_LAYERS:
                    return False
            # Mixed-precision gate: keep full-attention self_attn projections at FP16
            if MIXED_PRECISION_FULL_ATTN and ".self_attn." in tensor_name:
                li = _layer_index(tensor_name)
                if li is not None and LAYER_TYPES[li] == "full_attention":
                    return False
            return True
    return False


def is_rotatable_projection(tensor_name: str) -> bool:
    """A 2D residual-stream projection (q/k/v/o, in_proj_*, out_proj, gate/up/down) that
    must ALWAYS be Hadamard-rotated + norm-absorbed for transparency — even when we keep
    it FP16 (not ternarized). This is INDEPENDENT of the quantization toggles/layer gate:
    rotation governs correctness of the residual stream; quantization is a separate choice.
    Matches the base projection list only — embeddings, lm_head, norms, conv1d, biases do
    NOT match QUANTIZE_PATTERNS and are handled in the FP16 path instead."""
    return any(pattern in tensor_name for pattern in QUANTIZE_PATTERNS)


def get_layer_type(layer_idx: int) -> str:
    """Return 'full_attention' or 'linear_attention' for the given layer index."""
    if layer_idx < 0 or layer_idx >= NUM_HIDDEN_LAYERS:
        raise ValueError(f"Layer index {layer_idx} out of range [0, {NUM_HIDDEN_LAYERS})")
    return LAYER_TYPES[layer_idx]


@dataclass
class ConversionConfig:
    """Full configuration for a conversion run."""
    # Model source
    model_id: str = MODEL_ID
    model_cache_dir: Optional[str] = None

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    rotated_model_dir: Optional[Path] = None  # intermediate rotated checkpoint

    # Quantization
    block_size: int = BLOCK_SIZE
    apply_hadamard: bool = True    # Phase 1: Hadamard rotation before quantization
    rotation_only: bool = False    # Phase 1: rotation+norm absorption only, no ternary quantization
    use_calibration: bool = False  # Phase 2: calibration-aware quantization

    # Calibration (Phase 2)
    calibration_samples: int = 512
    calibration_datasets: list = field(default_factory=lambda: [
        "nvidia/Nemotron-Pretraining-Code-v1",
        "nvidia/Nemotron-Pretraining-Code-v2",
        "nvidia/Nemotron-CC-Math-v1",
    ])
    calibration_mix: dict = field(default_factory=lambda: {
        "nvidia/Nemotron-Pretraining-Code-v1": 0.35,
        "nvidia/Nemotron-Pretraining-Code-v2": 0.30,
        "nvidia/Nemotron-CC-Math-v1": 0.35,
    })

    # Hardware
    num_gpus: int = 2
    dtype: str = "bfloat16"  # source model precision

    # GGUF output
    gguf_format: str = "tq2_0"  # start with TQ2_0, then TQ1_0
    keep_vision: bool = True    # preserve vision encoder in FP16

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.rotated_model_dir:
            self.rotated_model_dir = Path(self.rotated_model_dir)
            self.rotated_model_dir.mkdir(parents=True, exist_ok=True)