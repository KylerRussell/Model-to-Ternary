"""Single source of truth for the 17-source calibration manifest.

Both `build_diverse_calib.py` (which reads the parquets) and `tools/fetch_data.py` (which downloads
them) import SOURCES from here, so the mixture fractions and the on-disk layout cannot drift apart.

Historically the manifest split sources into two kinds: 5 fetched from the Hub at build time ("hf")
and 12 read from a machine-local `CTM-Transformer/data_cache` tree ("ctm") that lived outside the
repo. That local tree was the one artifact with no provenance recorded anywhere — this module pins
the exact Hub repo + path for ALL 17 so the whole calibration set is reproducible from the repo alone.

LAYOUT: every source resolves to  $CTM_DATA/<dirname>/part_000000.parquet
`build_diverse_calib.chunks_from()` reads only the leading row-groups of that one file, and
`resolve()` takes `sorted(glob(...))[0]`, so exactly one part file per source is required.

GATING: 3 of the 5 Hub repos are gated (accept the license on the dataset page, then set HF_TOKEN).
Together they carry 0.52 of the mixture, so a token is not optional for a faithful rebuild.
"""

# (label, kind, locator, fraction) — `kind`/`locator` preserved verbatim for build_diverse_calib.
# fractions sum to 1.00
SOURCES = [
    ("CC-HighQuality",      "hf",  ("nvidia/Nemotron-CC-v2", "High-Quality/part_000000.parquet"), 0.14),
    ("CC-HQ-Synthetic",     "hf",  ("nvidia/Nemotron-CC-v2", "High-Quality-Synthetic/part_000000.parquet"), 0.11),
    ("CC-Diverse-QA",       "hf",  ("nvidia/Nemotron-CC-v2", "Diverse-QA/part_000000.parquet"), 0.11),
    ("CC-Math-4plus",       "hf",  ("nvidia/Nemotron-CC-Math-v1", "4plus/part_000000.parquet"), 0.06),
    ("Code-Synthetic",      "hf",  ("nvidia/Nemotron-Pretraining-Code-v1", "Synthetic-Code/part_000000.parquet"), 0.06),
    ("Wiki-Rewrite",        "ctm", "Nemotron-Pretraining-Wiki-Rewrite", 0.06),
    ("RQA",                 "ctm", "Nemotron-Pretraining-RQA", 0.05),
    ("STEM-SFT",            "ctm", "Nemotron-Pretraining-STEM-SFT", 0.05),
    ("InfiniByte-Reasoning","ctm", "Nemotron-Pretraining-InfiniByte-Reasoning", 0.05),
    ("Multiple-Choice",     "ctm", "Nemotron-Pretraining-Multiple-Choice", 0.05),
    ("Math-Textbooks",      "ctm", "Nemotron-Pretraining-Math-Textbooks", 0.04),
    ("Code-Concepts",       "ctm", "Nemotron-Pretraining-Code-Concepts", 0.04),
    ("4plus_MIND",          "ctm", "4plus_MIND", 0.04),
    ("Scientific-Coding",   "ctm", "Nemotron-Pretraining-Scientific-Coding", 0.04),
    ("Economics",           "ctm", "Nemotron-Pretraining-Economics", 0.04),
    ("Formal-Logic",        "ctm", "Nemotron-Pretraining-Formal-Logic", 0.03),
    ("Unconditional-Algo",  "ctm", "Nemotron-Pretraining-Unconditional-Algorithmic", 0.03),
]

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# UNIQUE-TOKEN CEILING: ~1.67B train tokens at these fractions.  (measured 2026-08-16)
#
# The mixture is capped by its SCARCEST source relative to that source's share. build_diverse_calib
# draws the held-out slice ON TOP of train (n_eval = n_train * eval_frac), so source i must supply
# frac_i * T * (1 + eval_frac) tokens, giving
#
#       T_max = min_i ( supply_i / (frac_i * (1 + eval_frac)) )
#
# BINDING SOURCE: Economics — 345,455 rows, 73.6M tokens EXACT (Qwen3.5-4B tokenizer, +1 eos/doc)
#                 73.6M / (0.04 * 1.1) = 1.67B
#
# This is a HARD ceiling, not an artifact of reading only part_000000. Four sources are SINGLE-PART
# upstream — one parquet IS the whole dataset, so they cannot be topped up by downloading more:
#
#   source               frac  parts  supply          caps T at
#   Economics            0.04    1    73.6M  (exact)  1.67B   <-- BINDING
#   Formal-Logic         0.03    1   129.4M  (exact)  3.92B
#   Unconditional-Algo   0.03    1   201.6M  (exact)  6.11B
#   Scientific-Coding    0.04    1   ~1.25B  (est)    ~28B
#
# Every other source has many parts upstream (RQA 175, CC-HQ-Synthetic 3216, CC-HighQuality 843, ...)
# and could be extended by fetching part_000001+, but that does not move T_max while Economics binds.
# Exact-counted runners-up, all far above the ceiling: Code-Concepts 122.4M -> 2.78B (61 parts),
# CC-Diverse-QA 500.9M -> 4.14B, CC-HighQuality 691.8M -> 4.49B, Multiple-Choice 443.6M -> 8.07B.
#
# Going beyond 1.67B REQUIRES cutting Economics' 0.04 share, which changes the mixture and breaks
# comparability with every recorded number (see RESULTS_SUMMARY §5 on the per-run held-out trap).
#
# For context: the old 60 GB box capped at 15.0M unique tokens (RESULTS_SUMMARY §9 / brief §4.3), so
# this is ~111x — about 6.8 doublings — against a measured +1.67 pt eval2k per doubling that showed
# no saturation at 16M. Data remains the least-exhausted lever in the project.
#
# CAUTION when re-deriving: sampling rows to estimate a source's supply is UNRELIABLE. A 1200-row
# sample put Economics at 109.5M, a 49% overestimate versus the exact 73.6M. Exact-count any source
# that could plausibly bind. Reproduce with the exact counter: iterate part_000000 in batches,
# sum len(tokenizer(text).input_ids) + 1 per doc.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
MAX_UNIQUE_TRAIN_TOKENS = 1_673_000_000   # at these fractions, eval_frac=0.10; binding: Economics
BINDING_SOURCE = "Economics"

# label -> (hub repo id, path within that repo). Verified against the live Hub file listings.
# The 12 former "ctm" folders were never a bespoke corpus — they are subsets of two OPEN Nemotron
# pretraining repos plus one gated one, which is why they were downloadable once and reproducible now.
HUB = {
    "CC-HighQuality":       ("nvidia/Nemotron-CC-v2", "High-Quality/part_000000.parquet"),
    "CC-HQ-Synthetic":      ("nvidia/Nemotron-CC-v2", "High-Quality-Synthetic/part_000000.parquet"),
    "CC-Diverse-QA":        ("nvidia/Nemotron-CC-v2", "Diverse-QA/part_000000.parquet"),
    "CC-Math-4plus":        ("nvidia/Nemotron-CC-Math-v1", "4plus/part_000000.parquet"),
    "Code-Synthetic":       ("nvidia/Nemotron-Pretraining-Code-v1", "Synthetic-Code/part_000000.parquet"),
    "Wiki-Rewrite":         ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Wiki-Rewrite/part_000000.parquet"),
    "RQA":                  ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-RQA/part_000000.parquet"),
    "STEM-SFT":             ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-STEM-SFT/part_000000.parquet"),
    "InfiniByte-Reasoning": ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-InfiniByte-Reasoning/part_000000.parquet"),
    "Math-Textbooks":       ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Math-Textbooks/part_000000.parquet"),
    "Scientific-Coding":    ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Scientific-Coding/part_000000.parquet"),
    "Multiple-Choice":      ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Multiple-Choice/part_000000.parquet"),
    "Code-Concepts":        ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Code-Concepts/part_000000.parquet"),
    "Economics":            ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Economics/part_000000.parquet"),
    "Formal-Logic":         ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Formal-Logic/part_000000.parquet"),
    "Unconditional-Algo":   ("nvidia/Nemotron-Pretraining-Specialized-v1.1", "Nemotron-Pretraining-Unconditional-Algorithmic/part_000000.parquet"),
    "4plus_MIND":           ("nvidia/Nemotron-CC-Math-v1", "4plus_MIND/part_000000.parquet"),
}

# Repos requiring an accepted license + HF_TOKEN. Checked live on 2026-08-16.
GATED_REPOS = {
    "nvidia/Nemotron-CC-v2",
    "nvidia/Nemotron-CC-Math-v1",
    "nvidia/Nemotron-Pretraining-Code-v1",
}


def dirname(label):
    """Local folder under $CTM_DATA for a source — the first path component of its Hub path.

    Chosen so the 12 former-"ctm" folders keep the exact names the old data_cache used
    (`Nemotron-Pretraining-RQA`, `4plus_MIND`, ...), which keeps any pre-existing local tree valid.
    """
    return HUB[label][1].split("/")[0]


def gated(label):
    return HUB[label][0] in GATED_REPOS


assert abs(sum(s[3] for s in SOURCES) - 1.0) < 1e-6, "fractions must sum to 1"
assert set(HUB) == {s[0] for s in SOURCES}, "HUB and SOURCES must cover the same labels"
