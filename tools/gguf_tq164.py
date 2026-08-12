#!/usr/bin/env python
"""gguf_tq164.py — rewrite an f16 GGUF into the TQ1_64 ternary format (1.7812 bpw).

Takes the f16 GGUF that `src/convert_hf_to_gguf_patched.py` already produces (so all architecture/KV/tensor-
naming logic is reused, not reimplemented) and re-encodes every tensor that is *actually* ternary into
TQ1_64. Everything else is copied through untouched.

WHY NOT `llama-quantize`: our weights are ALREADY exactly ternary x per-g64 scale. llama-quantize would
RE-DERIVE a quantization (at its own block size), which is exactly what destroyed the model when we exported
a g64 model to TQ2_0 (MMLU-Pro 25.7% -> 10.6%). This writer is lossless-by-construction: it reads the trits
that are there and packs them.

TENSOR SELECTION is EMPIRICAL, not name-based: a tensor is packed iff it is 2-D, its row length is a multiple
of 512, and every g64 group is on-grid (|w|/max(|w|) is either 0 or 1). That correctly catches
`token_embd.weight`, which IS ternary here (block_ap --quant-embed-head) but does NOT match the config's
QUANTIZE_PATTERNS — the name-based rule would have silently left 358 MB of ternary weights in q4_K.

  ./.venv/bin/python tools/gguf_tq164.py --in model-f16.gguf --out model-TQ1_64.gguf
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tq164 import encode_tensor, decode_row, QK_K, BLOCK_BYTES, GROUP, bpw  # noqa: E402

import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES  # noqa: E402

TQ1_64_ID = 43          # MUST match GGML_TYPE_TQ1_64 in the fork.
                        # NOT 42: that is already GGML_TYPE_Q2_0 upstream (ggml.h), and 36-38 are
                        # deprecated-but-reserved IQ4_NL_* slots. 43 is the first genuinely free id
                        # (GGML_TYPE_COUNT goes 43 -> 44).


def register_tq1_64():
    """Add TQ1_64 to the gguf package's enum + size table at runtime (the fork adds it properly in C)."""
    if "TQ1_64" in GGMLQuantizationType.__members__:
        return GGMLQuantizationType.__members__["TQ1_64"]
    m = int.__new__(GGMLQuantizationType, TQ1_64_ID)
    m._name_ = "TQ1_64"
    m._value_ = TQ1_64_ID
    GGMLQuantizationType._member_map_["TQ1_64"] = m
    GGMLQuantizationType._value2member_map_[TQ1_64_ID] = m
    GGMLQuantizationType._member_names_.append("TQ1_64")
    GGML_QUANT_SIZES[m] = (QK_K, BLOCK_BYTES)
    return m


def is_ternary_on_grid(w, group=GROUP, tol=1e-6):
    """True iff every g64 group is {-1,0,+1} x scale — i.e. the tensor really is ternary."""
    if w.ndim != 2 or w.shape[1] % QK_K:
        return False
    r = w.reshape(-1, group).astype(np.float32)
    mx = np.abs(r).max(axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    q = np.abs(r) / mx
    return bool(((q < tol) | (q > 1 - tol)).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="f16 GGUF from convert_hf_to_gguf_patched.py")
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--verify", action="store_true", help="decode a sample block back and check the round-trip")
    args = ap.parse_args()

    TQ = register_tq1_64()
    reader = gguf.GGUFReader(args.src, "r")
    arch = None
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = str(bytes(f.parts[f.data[0]]), "utf-8")
    if arch is None:
        raise SystemExit("could not read general.architecture from the source GGUF")
    writer = gguf.GGUFWriter(args.dst, arch)

    # ---- copy every KV field except the ones the writer sets itself ----
    # general.file_type must be RESTAMPED: copying it verbatim from the f16 source made the model
    # self-report as "F16" (llama-bench showed "qwen35 4B F16"), which is actively misleading.
    LLAMA_FTYPE_MOSTLY_TQ1_64 = 42      # llama.h enum (distinct from the ggml TYPE id 43)
    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
            "general.file_type"}
    n_kv = 0
    for f in reader.fields.values():
        if f.name in skip or not f.types:
            continue
        try:
            writer.add_key_value(f.name, f.contents(), f.types[0])
            n_kv += 1
        except Exception:
            pass  # writer-managed / unsupported field
    writer.add_file_type(LLAMA_FTYPE_MOSTLY_TQ1_64)
    print(f"copied {n_kv} KV fields (arch={arch}); file_type stamped TQ1_64", flush=True)

    # ---- tensors ----
    n_pack = n_copy = 0
    src_bytes = dst_bytes = 0
    for t in reader.tensors:
        w = np.array(t.data)
        src_bytes += w.nbytes
        # only F16/F32 source tensors can be inspected for ternary-ness
        if t.tensor_type in (GGMLQuantizationType.F16, GGMLQuantizationType.F32) and is_ternary_on_grid(w):
            # gguf expects the BYTE shape for a raw_dtype tensor and derives the element shape itself
            # (quant_shape_from_byte_shape), so pass [rows, bytes_per_row] and NO raw_shape.
            packed = encode_tensor(w.astype(np.float32)).reshape(w.shape[0], -1)
            writer.add_tensor(t.name, packed, raw_dtype=TQ)
            dst_bytes += packed.nbytes
            n_pack += 1
            if n_pack <= 3 or t.name.startswith(("token_embd", "output.")):
                print(f"  TQ1_64  {t.name:34s} {tuple(w.shape)} -> {packed.nbytes/1e6:8.1f} MB "
                      f"({w.nbytes/1e6:8.1f} MB f16)", flush=True)
        else:
            writer.add_tensor(t.name, w, raw_dtype=t.tensor_type)   # data is already in numpy order
            dst_bytes += w.nbytes
            n_copy += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out = Path(args.dst)
    print(f"\npacked {n_pack} tensors as TQ1_64 ({bpw():.4f} bpw), copied {n_copy} untouched")
    print(f"tensor bytes {src_bytes/1e6:.0f} MB -> {dst_bytes/1e6:.0f} MB   file: {out.stat().st_size/1e6:.0f} MB")

    if args.verify:
        rd = gguf.GGUFReader(args.dst, "r")
        for t in rd.tensors:
            if int(t.tensor_type) == TQ1_64_ID:
                blocks = np.array(t.data).reshape(-1, BLOCK_BYTES)
                cols = int(t.shape[0])
                rec = decode_row(blocks[: cols // QK_K], cols)
                print(f"verify: {t.name} first row decodes, "
                      f"{np.count_nonzero(rec)}/{cols} non-zero, max|w|={np.abs(rec).max():.6f}")
                break


if __name__ == "__main__":
    main()
