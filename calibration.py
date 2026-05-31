#!/usr/bin/env python3
"""
Calibration data loader for Phase 2 calibration-aware ternary quantization.

Loads a mixed sample from Nvidia Nemotron datasets:
  - Nemotron-Pretraining-Code-v1 (35%)
  - Nemotron-Pretraining-Code-v2 (30%)
  - Nemotron-CC-Math-v1 (35%)

Calibration data is used to compute Hessian matrices (H = X^T X) for
GPTQ-style quantization, which minimizes reconstruction error on
representative data.

Note: These datasets are gated on HuggingFace. You must:
  1. Log in: huggingface-cli login
  2. Accept the NVIDIA Data Agreement on each dataset page
"""

import argparse
import random
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Dataset configuration
CALIBRATION_MIX = {
    "nvidia/Nemotron-Pretraining-Code-v1": {
        "weight": 0.35,
        "text_field": "text",
        "split": "train",
        "description": "GitHub code in 11 languages",
    },
    "nvidia/Nemotron-Pretraining-Code-v2": {
        "weight": 0.30,
        "text_field": "text",
        "split": "train",
        "description": "Enhanced code corpus",
    },
    "nvidia/Nemotron-CC-Math-v1": {
        "weight": 0.35,
        "text_field": "text",
        "split": "train",
        "description": "Math from Common Crawl (LaTeX-formatted)",
    },
}

# Fallback datasets (no gating required)
FALLBACK_DATASETS = {
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "weight": 0.5,
        "text_field": "text",
        "split": "train",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "weight": 0.5,
        "text_field": "text",
        "split": "train",
        "streaming": True,
    },
}


def _load_local_dirs(parent_dir, tokenizer, total_samples, seq_length, text_field, seed,
                     shuffle_buffer=1000, max_rows_per_subset=50000):
    """Load + tokenize calibration text from a parent directory of HF datasets (e.g. your
    Nemotron-Pretraining-* subdirs). Uses STREAMING so the source size is irrelevant: it
    reads only enough rows to fill the calibration quota (plus a small shuffle buffer for
    randomness) and never materializes or caches the whole dataset. Mixes subdirs ~evenly.
    Auto-detects the text column from the first row; pass --text-field to override."""
    from datasets import load_dataset
    parent = Path(parent_dir)
    subdirs = [d for d in sorted(parent.iterdir())
               if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        subdirs = [parent]
    print(f"📚 Local calibration (streaming) from {parent} "
          f"({len(subdirs)} subsets), target {total_samples} samples")
    random.seed(seed)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    all_samples = []
    cand_fields = ("text", "content", "rewritten_text", "output", "completion",
                   "problem", "solution", "question", "answer")

    def open_stream(d):
        for loader in (
            lambda: load_dataset(str(d), split="train", streaming=True),
            lambda: load_dataset("parquet", data_files=str(d / "**" / "*.parquet"),
                                 split="train", streaming=True),
            lambda: load_dataset("json", data_files=str(d / "**" / "*.jsonl"),
                                 split="train", streaming=True),
            lambda: load_dataset("json", data_files=str(d / "**" / "*.json"),
                                 split="train", streaming=True),
        ):
            try:
                return loader()
            except Exception:
                continue
        return None

    for idx, d in enumerate(subdirs):
        if len(all_samples) >= total_samples:
            break
        ds = open_stream(d)
        if ds is None:
            print(f"   ⚠️ could not stream {d.name}; skipping")
            continue
        try:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        except Exception:
            pass  # some builders don't support shuffle; fall back to natural order

        it = iter(ds)
        try:
            first = next(it)
        except StopIteration:
            print(f"   ⚠️ {d.name} is empty; skipping")
            continue
        tf = text_field or next((c for c in cand_fields if c in first), None)
        if tf is None:
            tf = next((c for c in first if isinstance(first.get(c), str)), None)
        if tf is None:
            print(f"   ⚠️ no text field in {d.name} (cols={list(first)}); skipping")
            continue
        print(f"   {d.name}: streaming field '{tf}'")

        before = len(all_samples)
        # dynamic quota: split the REMAINING need across REMAINING subsets, so subsets
        # that fail/skip don't leave us short of total_samples.
        remaining_subsets = len(subdirs) - idx
        per_this = max(1, -(-(total_samples - before) // remaining_subsets))  # ceil-div
        target_here = min(before + per_this, total_samples)

        def rows():
            yield first
            yield from it

        buf = []           # running token buffer — pack across documents
        scanned = 0
        for ex in rows():
            if len(all_samples) >= target_here:
                break
            scanned += 1
            if scanned > max_rows_per_subset:
                print(f"      (scanned {max_rows_per_subset} rows; taking "
                      f"{len(all_samples) - before} and moving on)")
                break
            if scanned % 5000 == 0:
                print(f"      ...scanned {scanned} rows, "
                      f"{len(all_samples) - before}/{per_this} samples so far")
            txt = ex.get(tf) or ""
            if len(txt) < 50:
                continue
            buf.extend(tokenizer.encode(txt, add_special_tokens=False))
            if eos_id is not None:
                buf.append(eos_id)
            while len(buf) >= seq_length and len(all_samples) < target_here:
                all_samples.append(buf[:seq_length])
                del buf[:seq_length]

    random.shuffle(all_samples)
    all_samples = all_samples[:total_samples]
    print(f"   ✅ {len(all_samples)} samples ({len(all_samples) * seq_length:,} tokens)")
    return all_samples


def load_calibration_data(
    total_samples: int = 512,
    seq_length: int = 2048,
    tokenizer_id: str = "Qwen/Qwen3.6-27B",
    seed: int = 42,
    use_fallback: bool = False,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    text_field: Optional[str] = None,
) -> list:
    """
    Load and tokenize calibration samples.

    Returns a list of tokenized sequences, each of length `seq_length`.
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required: pip install datasets")
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required: pip install transformers")

    random.seed(seed)
    print(f"📚 Loading calibration data ({total_samples} samples, {seq_length} tokens each)")

    # Load tokenizer
    print(f"   Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Local domain data (e.g. your Nemotron data_cache) takes priority when given.
    if local_dir:
        return _load_local_dirs(local_dir, tokenizer, total_samples, seq_length,
                                text_field, seed)

    datasets_config = FALLBACK_DATASETS if use_fallback else CALIBRATION_MIX
    all_samples = []

    for ds_id, ds_cfg in datasets_config.items():
        n_samples = int(total_samples * ds_cfg["weight"])
        text_field = ds_cfg["text_field"]
        path = ds_cfg.get("path", ds_id)
        name = ds_cfg.get("name")
        print(f"\n   Loading {n_samples} samples from: {path} (config: {name})")

        try:
            if ds_cfg.get("streaming", False):
                ds = load_dataset(
                    path,
                    name=name,
                    split=ds_cfg["split"],
                    streaming=True,
                    cache_dir=cache_dir,
                )
                # Take samples from stream
                texts = []
                for i, example in enumerate(ds):
                    if len(all_samples) >= total_samples or len(texts) >= n_samples * 100:
                        break
                    text = example.get(text_field, "")
                    if len(text) > 100:  # skip very short samples
                        texts.append(text)
            else:
                ds = load_dataset(
                    path,
                    name=name,
                    split=ds_cfg["split"],
                    cache_dir=cache_dir,
                )
                # Random sample
                indices = random.sample(range(len(ds)), min(n_samples * 3, len(ds)))
                texts = [ds[i][text_field] for i in indices if ds[i].get(text_field)]

        except Exception as e:
            print(f"   ⚠️ Failed to load {ds_id}: {e}")
            if not use_fallback:
                print(f"   Tip: Run 'huggingface-cli login' and accept the dataset agreement")
            continue

        # Tokenize and chunk into seq_length sequences
        for text in texts:
            if len(all_samples) >= total_samples:
                break
            tokens = tokenizer.encode(text, add_special_tokens=False)
            # Split into chunks of seq_length
            for start in range(0, len(tokens) - seq_length + 1, seq_length):
                if len(all_samples) >= total_samples:
                    break
                chunk = tokens[start:start + seq_length]
                if len(chunk) == seq_length:
                    all_samples.append(chunk)

    # Shuffle
    random.shuffle(all_samples)
    all_samples = all_samples[:total_samples]

    print(f"\n   ✅ Loaded {len(all_samples)} calibration samples "
          f"({len(all_samples) * seq_length:,} tokens total)")

    return all_samples


def save_calibration_cache(samples: list, path: Path):
    """Save tokenized calibration data to disk for reuse."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(samples, f)
    print(f"   💾 Calibration data cached to: {path}")


def load_calibration_cache(path: Path) -> Optional[list]:
    """Load cached calibration data."""
    import json
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Load calibration data for ternary quantization")
    parser.add_argument("--samples", type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length per sample")
    parser.add_argument("--fallback", action="store_true",
                        help="Use fallback datasets (no gating)")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache directory")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Parent dir of local datasets (e.g. your Nemotron data_cache). "
                             "Subdirectories are loaded and mixed; takes priority over --fallback.")
    parser.add_argument("--text-field", type=str, default=None,
                        help="Override the text column name (auto-detected if omitted).")
    parser.add_argument("--output", type=str, default="./output/calibration_data.json",
                        help="Path to save calibration data")
    args = parser.parse_args()

    samples = load_calibration_data(
        total_samples=args.samples,
        seq_length=args.seq_length,
        use_fallback=args.fallback,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        text_field=args.text_field,
    )

    save_calibration_cache(samples, Path(args.output))


if __name__ == "__main__":
    main()