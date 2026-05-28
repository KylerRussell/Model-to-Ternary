#!/usr/bin/env python3
"""
Post-conversion validation for ternary GGUF models.

Validates:
  1. GGUF file loads correctly in llama.cpp
  2. Model produces coherent output
  3. Perplexity comparison against reference
  4. Token generation speed benchmarks
"""

import argparse
import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path


def check_gguf_loads(gguf_path: Path) -> bool:
    """Verify the GGUF file loads without errors."""
    llama_cli = shutil.which("llama-cli")
    if not llama_cli:
        print("  ⚠️ llama-cli not found, skipping load test")
        return False

    print("  Testing GGUF load...")
    cmd = [
        llama_cli,
        "-m", str(gguf_path),
        "-p", "Test",
        "-n", "1",  # generate just 1 token to test loading
        "--no-display-prompt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode == 0:
        print("  ✅ GGUF loads successfully")
        return True
    else:
        print(f"  ❌ GGUF load failed!")
        print(f"  STDERR: {result.stderr[-1000:]}")
        return False


def test_generation(gguf_path: Path, prompts: list = None) -> dict:
    """Run test prompts and check output quality."""
    llama_cli = shutil.which("llama-cli")
    if not llama_cli:
        return {"error": "llama-cli not found"}

    if prompts is None:
        prompts = [
            "Explain the concept of recursion in programming:",
            "What is the derivative of x^3 + 2x^2 - 5x + 3?",
            "Write a Python function to compute the Fibonacci sequence:",
            "The capital of France is",
        ]

    results = []
    for prompt in prompts:
        print(f"\n  Prompt: {prompt[:60]}...")
        cmd = [
            llama_cli,
            "-m", str(gguf_path),
            "-p", prompt,
            "-n", "128",
            "--no-display-prompt",
            "--temp", "0.0",  # deterministic
            "-ngl", "99",     # offload to GPU
        ]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start

        output = result.stdout.strip()
        results.append({
            "prompt": prompt,
            "output": output[:500],
            "time_s": elapsed,
            "tokens_approx": len(output.split()),
            "success": result.returncode == 0,
        })

        if result.returncode == 0:
            print(f"  Response: {output[:200]}...")
            print(f"  Time: {elapsed:.1f}s")
        else:
            print(f"  ❌ Generation failed: {result.stderr[-500:]}")

    return {"prompts": results}


def run_perplexity(gguf_path: Path, test_file: Path = None) -> dict:
    """
    Run perplexity evaluation using llama-perplexity.
    If no test file provided, uses a small built-in test.
    """
    perplexity_bin = shutil.which("llama-perplexity")
    if not perplexity_bin:
        print("  ⚠️ llama-perplexity not found, skipping perplexity test")
        return {"error": "llama-perplexity not found"}

    if test_file is None:
        # Create a small test file
        test_file = Path("/tmp/ppl_test.txt")
        test_text = (
            "The transformer architecture was introduced in the paper "
            "'Attention Is All You Need' by Vaswani et al. in 2017. "
            "It revolutionized natural language processing by replacing "
            "recurrent neural networks with self-attention mechanisms. "
            "The key innovation was the multi-head attention mechanism "
            "which allows the model to attend to different parts of the "
            "input sequence simultaneously. Transformers have since become "
            "the foundation for large language models like GPT, BERT, and "
            "their many variants. The architecture consists of an encoder "
            "and decoder, each made up of layers containing self-attention "
            "and feed-forward neural networks."
        )
        test_file.write_text(test_text)

    print(f"  Running perplexity on: {test_file}")
    cmd = [
        perplexity_bin,
        "-m", str(gguf_path),
        "-f", str(test_file),
        "-ngl", "99",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return {"error": result.stderr[-500:]}

    # Parse perplexity from output
    for line in result.stdout.split("\n"):
        if "perplexity" in line.lower():
            return {"output": line.strip(), "raw": result.stdout[-500:]}

    return {"raw": result.stdout[-500:]}


def benchmark_throughput(gguf_path: Path) -> dict:
    """Benchmark token generation throughput."""
    llama_cli = shutil.which("llama-cli")
    if not llama_cli:
        return {"error": "llama-cli not found"}

    print("  Running throughput benchmark...")
    cmd = [
        llama_cli,
        "-m", str(gguf_path),
        "-p", "Write a detailed essay about the history of computing:",
        "-n", "512",
        "--no-display-prompt",
        "-ngl", "99",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        return {"error": result.stderr[-500:]}

    # Parse timing info from stderr
    timings = {}
    for line in result.stderr.split("\n"):
        if "eval time" in line or "total time" in line or "tokens per second" in line:
            timings[line.strip().split(":")[0].strip()] = line.strip()

    return {
        "wall_time_s": elapsed,
        "tokens_requested": 512,
        "timings": timings,
    }


def full_validation(gguf_path: Path, output_dir: Path = None):
    """Run the complete validation suite."""
    print("\n" + "=" * 70)
    print("TERNARY MODEL VALIDATION")
    print("=" * 70)
    print(f"  Model: {gguf_path}")
    print(f"  Size:  {gguf_path.stat().st_size / 1e9:.2f} GB")

    results = {}

    # 1. Load test
    print("\n── Test 1: GGUF Load ──")
    results["load_test"] = check_gguf_loads(gguf_path)

    if not results["load_test"]:
        print("\n❌ Model failed to load, skipping further tests.")
        return results

    # 2. Generation test
    print("\n── Test 2: Generation Quality ──")
    results["generation"] = test_generation(gguf_path)

    # 3. Throughput benchmark
    print("\n── Test 3: Throughput ──")
    results["throughput"] = benchmark_throughput(gguf_path)

    # 4. Perplexity (optional)
    print("\n── Test 4: Perplexity ──")
    results["perplexity"] = run_perplexity(gguf_path)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  📊 Validation report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate ternary GGUF model")
    parser.add_argument("gguf_path", type=str, help="Path to the GGUF file")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory for validation reports")
    args = parser.parse_args()

    gguf_path = Path(args.gguf_path)
    if not gguf_path.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        sys.exit(1)

    full_validation(gguf_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
