#!/usr/bin/env python
"""Populate the in-repo `data/` folder with the 17-source calibration corpus.

Replaces the old machine-local `~/Documents/CTM-Transformer/data_cache` tree, so a fresh box needs
only this repo + an HF token. Downloads exactly one `part_000000.parquet` per source (that is all
build_diverse_calib.py ever reads) into `data/<dirname>/`.

  python tools/fetch_data.py                 # fetch everything that is accessible
  python tools/fetch_data.py --check         # report presence/size, download nothing
  python tools/fetch_data.py --open-only     # skip the 3 gated repos

GATED SOURCES: nvidia/Nemotron-CC-v2, nvidia/Nemotron-CC-Math-v1, nvidia/Nemotron-Pretraining-Code-v1
carry 0.52 of the mixture. Accept each license on its dataset page, then `export HF_TOKEN=hf_...`
(or `hf auth login`). Without them the calib is buildable but is NOT the documented mixture, and
results are not comparable to the recorded numbers.
"""
import argparse, os, sys, shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from calib_sources import SOURCES, HUB, dirname, gated  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.environ.get("CTM_DATA") or os.path.join(REPO, "data"))
ap.add_argument("--check", action="store_true", help="report only, download nothing")
ap.add_argument("--open-only", action="store_true", help="skip repos that require a token")
ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
a = ap.parse_args()

from huggingface_hub import hf_hub_download, get_token  # noqa: E402
from huggingface_hub.errors import GatedRepoError  # noqa: E402

# --token > HF_TOKEN > the token stored by `hf auth login`. Report the one that will actually be used:
# hf_hub_download(token=None) already falls back to the stored token, so "HF_TOKEN unset" alone does
# not mean the gated sources will fail.
_tok_src = ("--token" if a.token and a.token != os.environ.get("HF_TOKEN") else
            "HF_TOKEN" if a.token else
            "stored login" if get_token() else None)

os.makedirs(a.data_dir, exist_ok=True)
print(f"data dir: {a.data_dir}")
print(f"HF token: {_tok_src or 'NONE — gated sources (0.52 of the mixture) will fail'}\n")

have = miss = 0
have_frac = 0.0
missing = []

for label, kind, loc, frac in SOURCES:
    repo, path = HUB[label]
    dest_dir = os.path.join(a.data_dir, dirname(label))
    dest = os.path.join(dest_dir, "part_000000.parquet")
    tag = "GATED" if gated(label) else "open "

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [have] {label:22s} {tag} {os.path.getsize(dest)/1e6:8.1f} MB")
        have += 1; have_frac += frac
        continue

    if a.check:
        print(f"  [MISS] {label:22s} {tag} frac {frac:.2f}  <- {repo}/{path}")
        miss += 1; missing.append(label)
        continue

    if a.open_only and gated(label):
        print(f"  [skip] {label:22s} {tag} frac {frac:.2f}  (--open-only)")
        miss += 1; missing.append(label)
        continue

    try:
        print(f"  [get ] {label:22s} {tag} frac {frac:.2f}  <- {repo}/{path}", flush=True)
        src = hf_hub_download(repo, path, repo_type="dataset", token=a.token)
        os.makedirs(dest_dir, exist_ok=True)
        # copy (not symlink) out of the HF cache so `data/` is self-contained and survives a cache purge
        shutil.copyfile(src, dest)
        print(f"         -> {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")
        have += 1; have_frac += frac
    except GatedRepoError:
        print(f"         !! GATED: accept the license at https://huggingface.co/datasets/{repo}"
              f" then set HF_TOKEN")
        miss += 1; missing.append(label)
    except Exception as e:
        print(f"         !! FAILED: {type(e).__name__}: {str(e)[:120]}")
        miss += 1; missing.append(label)

print(f"\n{have}/{len(SOURCES)} sources present — {have_frac:.2f} of the calibration mixture by weight")
if missing:
    print(f"missing: {', '.join(missing)}")
    print("build_diverse_calib.py will FAIL on a missing source; fetch them or the mixture is not the "
          "documented one and results are not comparable.")
sys.exit(0 if not missing else 1)
