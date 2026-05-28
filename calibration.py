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
        "name": "wikitext-2-raw-v1",
        "weight": 0.5,
        "text_field": "text",
        "split": "train",
    },
    "c4": {
        "name": "allenai/c4",
        "weight": 0.5,
        "text_field": "text",
        "split": "train",
        "streaming": True,
    },
}


def load_calibration_data(
    total_samples: int = 512,
    seq_length: int = 2048,
    tokenizer_id: str = "Qwen/Qwen3.6-27B",
    seed: int = 42,
    use_fallback: bool = False,
    cache_dir: Optional[str] = None,
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

    datasets_config = FALLBACK_DATASETS if use_fallback else CALIBRATION_MIX
    all_samples = []

    for ds_id, ds_cfg in datasets_config.items():
        n_samples = int(total_samples * ds_cfg["weight"])
        text_field = ds_cfg["text_field"]
        print(f"\n   Loading {n_samples} samples from: {ds_id}")

        try:
            if ds_cfg.get("streaming", False):
                ds = load_dataset(
                    ds_cfg.get("name", ds_id),
                    split=ds_cfg["split"],
                    streaming=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                # Take samples from stream
                texts = []
                for i, example in enumerate(ds):
                    if i >= n_samples * 3:  # over-sample to filter short ones
                        break
                    text = example.get(text_field, "")
                    if len(text) > 100:  # skip very short samples
                        texts.append(text)
            else:
                ds = load_dataset(
                    ds_cfg.get("name", ds_id),
                    split=ds_cfg["split"],
                    cache_dir=cache_dir,
                    trust_remote_code=True,
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
    parser.add_argument("--output", type=str, default="./output/calibration_data.json",
                        help="Path to save calibration data")
    args = parser.parse_args()

    samples = load_calibration_data(
        total_samples=args.samples,
        seq_length=args.seq_length,
        use_fallback=args.fallback,
        cache_dir=args.cache_dir,
    )

    save_calibration_cache(samples, Path(args.output))


if __name__ == "__main__":
    main()
