#!/usr/bin/env python
"""Apply the two zero-risk memory wins to src/e2e_qp_distill.py (report Stage-0, rows 2 and 3).

  row 2: 8-bit block-wise Adam states for the CPU-resident latents   -6.0 B/latent
  row 3: best-checkpoint snapshot evicted to NVMe (file-backed)      -4.0 B/latent
                                                                     ---------------
                                                        27.3 -> ~17.3 B/latent

Both are OPT-IN (`--adam8bit`, `--snap-nvme`), so the default code path is untouched.

DO NOT RUN WHILE A MULTI-STAGE SCRIPT IS EXECUTING: run_groupjoint.sh et al. spawn a FRESH python per
stage, so editing this file mid-run silently changes the code under later stages.

Measured before writing (see the session log):
  * quantiser round-trip max rel err 0.00394 (int8 => 1/254, as designed)
  * Adam tracking: total displacement ratio 0.9931 vs fp32 over 400 steps
  * FLIP DECISIONS -- the thing that actually matters -- 99.848% identical at a ~10% flip rate,
    99.630% at ~24%, with a slight bias toward FEWER flips that TALR servos away automatically
  * speed: 1.43x per-tensor Adam, but only +10.3% on the full training step
  * memmap snapshot: anonymous RSS unchanged (0.28 GB) where an in-RAM clone cost 0.75 GB
"""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "src" / "e2e_qp_distill.py"

EDITS = [
    # ── A. import the CPU 8-bit Adam helpers ────────────────────────────────────────────────────
    (
        "        _adam_scratch = {\"b\": None}          # one reusable denom buffer for the whole run (see _adam_step_one)",
        "        _adam8bit = bool(getattr(args, \"adam8bit\", False))\n"
        "        if _adam8bit:\n"
        "            from adam8bit_cpu import Adam8bitState, adam_step_8bit\n"
        "            log(f\"   8-bit block-wise Adam states for {len(latents)} latents \"\n"
        "                f\"(saves {sum(p_.numel() for p_ in latents)*6/1e9:.1f}GB; moments only -- the LATENT\"\n"
        "                f\" stays fp32 because the STE gate needs boundary resolution)\")\n"
        "        _adam_scratch = {\"b\": None}          # one reusable denom buffer for the whole run (see _adam_step_one)",
    ),
    # ── B. branch the custom Adam step onto the quantised path ──────────────────────────────────
    (
        "            st = opt.state[param]\n"
        "            if len(st) == 0:\n"
        "                i = _lat_index.get(id(param), 0)",
        "            st = opt.state[param]\n"
        "            if _adam8bit:\n"
        "                # Only the MOMENTS are quantised. Measured 99.85% identical flip decisions vs fp32.\n"
        "                if \"q\" not in st:\n"
        "                    st[\"q\"] = Adam8bitState(param.numel())\n"
        "                adam_step_8bit(param, param.grad, st[\"q\"], group[\"lr\"],\n"
        "                               group[\"betas\"], group[\"eps\"], scratch=_adam_scratch)\n"
        "                return\n"
        "            if len(st) == 0:\n"
        "                i = _lat_index.get(id(param), 0)",
    ),
    # ── C. evict the best-checkpoint snapshot for LATENTS to a memmap ───────────────────────────
    (
        "    def _snap():\n"
        "        if _snap_bufs[\"s\"] is None:\n"
        "            _snap_bufs[\"s\"] = [s.detach().clone() for s in scales]",
        "    # The snapshot holds a full copy of `scales`, which INCLUDES the latents -- 4 B/latent of pure\n"
        "    # anonymous RSS that exists only to restore a stage's best checkpoint. Backing it with a memmap\n"
        "    # makes those pages reclaimable page cache instead (measured: anon RSS unchanged vs +0.75GB for\n"
        "    # an in-RAM clone), so it stops counting against the cgroup limit that actually kills runs.\n"
        "    _snap_lat_ids = {id(p_) for p_ in latents}\n"
        "    _snap_dir = None\n"
        "    if getattr(args, \"snap_nvme\", False) and latents:\n"
        "        _snap_dir = Path(args.out).parent / \"_snap_state\"\n"
        "        shutil.rmtree(_snap_dir, ignore_errors=True)\n"
        "        _snap_dir.mkdir(parents=True, exist_ok=True)\n"
        "        log(f\"   best-checkpoint snapshot NVMe-backed at {_snap_dir} \"\n"
        "            f\"(keeps {sum(p_.numel() for p_ in latents)*4/1e9:.1f}GB off anonymous RSS)\")\n"
        "\n"
        "    def _snap_alloc(t, i):\n"
        "        if _snap_dir is None or id(t) not in _snap_lat_ids or t.device.type != \"cpu\":\n"
        "            return t.detach().clone()\n"
        "        import numpy as _np\n"
        "        arr = _np.memmap(str(_snap_dir / f\"s_{i}.dat\"), dtype=_np.float32, mode=\"w+\",\n"
        "                         shape=(t.numel(),))\n"
        "        buf = torch.from_numpy(arr).view_as(t)\n"
        "        buf.copy_(t.detach())\n"
        "        return buf\n"
        "\n"
        "    def _snap():\n"
        "        if _snap_bufs[\"s\"] is None:\n"
        "            _snap_bufs[\"s\"] = [_snap_alloc(s, i) for i, s in enumerate(scales)]",
    ),
    # ── D. flags ────────────────────────────────────────────────────────────────────────────────
    (
        "    ap.add_argument(\"--latent-lr-ramp-steps\", type=int, default=0,",
        "    ap.add_argument(\"--adam8bit\", action=\"store_true\",\n"
        "                    help=\"Block-wise 8-bit Adam MOMENTS for the latents (-6 B/latent, +~10%% step time). \"\n"
        "                         \"The latent itself stays fp32 -- bf16 latents were measured to destroy the STE \"\n"
        "                         \"gate's boundary resolution (KL 0.9495 vs 0.6624). Flip decisions measured \"\n"
        "                         \"99.85%% identical to fp32 moments.\")\n"
        "    ap.add_argument(\"--snap-nvme\", action=\"store_true\",\n"
        "                    help=\"Back the best-checkpoint snapshot of the latents with a memmap (-4 B/latent of \"\n"
        "                         \"ANONYMOUS rss; the pages become reclaimable page cache).\")\n"
        "    ap.add_argument(\"--latent-lr-ramp-steps\", type=int, default=0,",
    ),
]


def main(check_only=False):
    src = P.read_text()
    if "--adam8bit" in src:
        print("  already applied — nothing to do"); return 0
    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        if old not in out:
            print(f"  FAILED: anchor {i} not found — file has changed, patch by hand"); return 1
        if out.count(old) != 1:
            print(f"  FAILED: anchor {i} matches {out.count(old)}x — ambiguous"); return 1
        out = out.replace(old, new, 1)
    import ast
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"  FAILED: result does not parse — {e}"); return 1
    if check_only:
        print("  dry run OK: all 4 anchors unique, result parses"); return 0
    P.write_text(out)
    print("  applied 4 edits; result parses. Enable with --adam8bit --snap-nvme")
    return 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check" in sys.argv))
