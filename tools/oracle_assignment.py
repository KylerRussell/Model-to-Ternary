#!/usr/bin/env python
"""oracle_assignment.py — ORACLE test for the ASSIGNMENT lever (researcher report 2026-08-05, Q5).

QUESTION: at FIXED deploy size (g64, 1.781 bpw), how much of the FP gap can better trit ASSIGNMENTS recover?
This is the size-free lever, so per the corrected framing it is the thesis-aligned one — the granularity lever
costs bpw (g32-Q8=1.906) and is off-budget, but assignments are free.

METHOD (offline, no training): freeze the model; take a FRESH gradient of the held-out CE+KL loss w.r.t. the
effective down_proj weights; rank every candidate trit change by its first-order predicted gain
Δ̂ = ḡ·s·(t'−t); commit the top-K (UNCONSTRAINED — no trust region, no per-block cap); measure the ACTUAL
held-out loss + FP-agreement; revert; repeat for a sweep of K. That gives the ROI curve of the assignment
lever. Reuses the existing Arm-B machinery (mutable int8 trits + gradient accumulator); eta→∞ reduces the
proximal gate to pure first-order top-K.

READING (report thresholds, on fraction of the FP gap recovered):
  >=50%  → ASSIGNMENT-LIMITED: good assignments exist, we just can't reach them with an uncontrolled STE.
           Build the flip-control machinery (flip budget + Arm-B under CE+KL + V/P alternation). It's free size.
  <=20%  → the assignment lever is nearly exhausted; the residual is capacity — and since finer scales are
           off-budget (>1.75 bpw), that means accepting a sub-FP ceiling at this size.
TWO REASONS THIS IS A LOWER BOUND (read a null result cautiously):
  1. Scales are held FIXED while trits flip; a real run would refit scales after each flip batch (the P-phase),
     which only helps.
  2. Flips are ranked by a FIRST-ORDER estimate, but a trit flip is a LARGE discrete step (Δw = s·d) and K
     simultaneous flips are assumed non-interacting. This is a first-order greedy heuristic, NOT the report's
     full oracle (AdaRound relaxation / greedy OBQ WITH compensation).
⇒ A LARGE gain is strong evidence of assignment-limited. A SMALL gain is WEAK evidence of capacity-limited —
  it may only mean first-order ranking is a poor search. Escalate to AdaRound/OBQ before concluding "capacity".

  ORIG_MODEL=<orig-config> ./.venv/bin/python tools/oracle_assignment.py \
      --student output_4bpipe/e2eqp/modified_model --calib output_4bpipe/calibration_data.json \
      --teacher-cache output_4bpipe/teacher_topk.pt --ce-weight 0.5 --probe-n 16 --eval-n 32
"""
import argparse, os, sys, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from e2e_qp_distill import (build_student, load_calib_batches, BLOCK_SIZE, TernaryScaleLinear,
                            chunked_hidden_state_loss, chunked_ce, _chunked_argmax)


@torch.no_grad()
def oracle_flip(mods, frac):
    """Commit the top-`frac` trit changes by FIRST-ORDER predicted gain Δ̂ = ḡ·s·(t'−t); return a revert list.

    NOT arm_b_gated_flip: with its trust region removed (eta→∞) its gate `Δ<0` qualifies ~half of all weights,
    so its candidate list would be GBs. Here the budget is applied PER MODULE via kthvalue on that module's
    gain tensor — memory-safe, and an equal-per-layer split is a fine approximation for an ROI curve."""
    revert = []
    committed = 0
    for m in mods:
        s = m.scale.detach().clamp_min(1e-8).unsqueeze(1)            # [n_blocks,1]
        g = m.flip_grad.float()                                      # ∂loss/∂deq
        t = m.tern_b.float()
        best = torch.full_like(g, float("inf")); best_new = t.clone()
        for d in (-2, -1, 1, 2):                                     # reachable trit deltas
            sd = s * d
            gain = torch.where((t + d >= -1) & (t + d <= 1), g * sd, torch.full_like(g, float("inf")))
            better = gain < best
            best = torch.where(better, gain, best); best_new = torch.where(better, t + d, best_new)
        k = int(frac * best.numel())
        if k < 1:
            del best, best_new, g, t; continue
        flat = best.reshape(-1)
        thresh = flat.kthvalue(min(k, flat.numel())).values
        sel = (best <= thresh) & (best < 0)                          # only genuinely-predicted-improving flips
        if bool(sel.any()):
            idx = sel.nonzero(as_tuple=False)
            b, c = idx[:, 0], idx[:, 1]
            revert.append((m, (b, c), m.tern_b[b, c].clone()))
            m.tern_b[b, c] = best_new[sel].to(torch.int8)
            committed += int(sel.sum())
        del best, best_new, g, t, sel
    return committed, revert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--orig-config-path", default=os.environ.get("ORIG_MODEL"))
    ap.add_argument("--calib", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--scope", default="down", choices=["down", "mlp", "all"])
    ap.add_argument("--ce-weight", type=float, default=0.5)
    ap.add_argument("--ce-positions", type=int, default=512)
    ap.add_argument("--probe-n", type=int, default=16, help="#seqs for the fresh gradient probe")
    ap.add_argument("--eval-n", type=int, default=32, help="#held-out seqs scored per K (fixed slice)")
    ap.add_argument("--seq", type=int, default=2560)
    ap.add_argument("--budgets", default="1e-6,1e-5,1e-4,1e-3,1e-2,5e-2")
    ap.add_argument("--out", default="logs/oracle_assignment.json")
    args = ap.parse_args()
    dev = "cuda:0"

    print(f"loading student {args.student} ...", flush=True)
    core, _ = build_student(args.student, args.orig_config_path, BLOCK_SIZE, dev)
    core.config.use_cache = False
    for p in core.parameters():
        p.requires_grad_(False)

    def in_scope(name):
        if args.scope == "down": return name.endswith("down_proj")
        if args.scope == "mlp":  return "mlp" in name
        return True
    mods = [m for n, m in core.named_modules() if isinstance(m, TernaryScaleLinear) and in_scope(n)]
    for m in mods:
        m.enable_arm_b()          # mutable int8 trits + bf16 gradient accumulator (no fp32 latents)
        # deq = tern_b * scale; the flip_grad hook only attaches when deq.requires_grad, so the scale must
        # carry grad even though we never step it (scales stay FIXED — this is the conservative oracle).
        m.scale.requires_grad_(True)
    ntot = sum(m.tern_b.numel() for m in mods)
    print(f"  scope={args.scope}: {len(mods)} linears, {ntot/1e6:.0f}M trits mutable", flush=True)

    # post-norm hidden capture + CPU lm_head weight (chunked losses ⇒ never materialise [T,V])
    base = core.model.language_model if hasattr(core.model, "language_model") else core.model
    hid = {}
    base.norm.register_forward_hook(lambda mod, i, o: hid.__setitem__("h", o))
    with torch.no_grad():
        w = core.lm_head.dequant() if hasattr(core.lm_head, "dequant") else core.lm_head.weight
        Wlm = w.detach().to(torch.bfloat16).cpu().contiguous()
        del w
    V = Wlm.shape[0]
    core.lm_head.forward = (lambda x, _v=V: x.new_zeros(x.shape[:-1] + (_v,)))   # unused; avoid 2.4GB unpack
    torch.cuda.empty_cache()

    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")
    cache = torch.load(args.teacher_cache, map_location="cpu")
    n_avail = min(len(batches), len(cache["idx"]))      # cache stores per-seq LISTS, not stacked tensors
    # disjoint, FIXED slices from the TAIL (the trainer's held-out convention) — same seqs for every K
    ev_idx = list(range(n_avail - args.eval_n, n_avail))
    pr_idx = list(range(n_avail - args.eval_n - args.probe_n, n_avail - args.eval_n))
    print(f"  probe {len(pr_idx)} seqs | eval {len(ev_idx)} seqs (fixed, disjoint)", flush=True)

    def losses(idx_list, want_agree=False):
        """mean held-out CE+KL (and optionally FP-agreement%) over a fixed slice, no grad."""
        tk = ta = 0.0; agree = 0; ntok = 0
        core.eval()
        with torch.no_grad():
            for bi in idx_list:
                ids = batches[bi].to(dev)
                ti = cache["idx"][bi].unsqueeze(0).to(dev); tv = cache["val"][bi].unsqueeze(0).to(dev)
                hid.clear(); core(ids, logits_to_keep=1); h = hid["h"]
                kl = float(chunked_hidden_state_loss(h, Wlm, ti, tv, temperature=1.0,
                                                     loss_type="cakld", chunk_size=4096))
                ce = 0.0
                if args.ce_weight > 0:
                    H = h[0, :-1, :]; tg = ids[0, 1:]
                    n = args.ce_positions
                    if 0 < n < H.shape[0]:
                        sel = torch.arange(0, H.shape[0], max(1, H.shape[0] // n), device=H.device)[:n]
                        H, tg = H[sel], tg[sel]          # DETERMINISTIC stride (not random) so K's compare
                    ce = float(chunked_ce(H, Wlm, tg))
                tk += kl; ta += ce
                if want_agree:
                    am = _chunked_argmax(h[0, :-1, :], Wlm, chunk=4096)
                    agree += int((am == ti[0, :-1, 0]).sum()); ntok += h.shape[1] - 1
        n = max(1, len(idx_list))
        return tk / n, ta / n, (100.0 * agree / max(1, ntok) if want_agree else None)

    # ── baseline ──
    b_kl, b_ce, b_ag = losses(ev_idx, want_agree=True)
    b_tot = b_kl + args.ce_weight * b_ce
    print(f"\n  BASELINE: KL {b_kl:.4f} | CE {b_ce:.4f} | CE+KL {b_tot:.4f} | FP-agreement {b_ag:.2f}%")
    print(f"  FP gap to close = {100.0 - b_ag:.2f} points\n", flush=True)

    # ── fresh gradient probe of CE+KL at frozen (T,s) ──
    # Gradient checkpointing is MANDATORY here: the probe is the only part that runs backward, and a 2560-token
    # 4B backward without it stores ~17GB of activations on top of flip_grad (1.5GB) + tern_b (0.75GB) + model
    # (2.6GB) ⇒ OOM. (losses() runs under no_grad, which is why the baseline succeeds either way.)
    try:
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("  gradient checkpointing ON for the probe", flush=True)
    except Exception as e:
        print(f"  WARNING: could not enable gradient checkpointing ({e}) — probe may OOM", flush=True)
    print("  probing fresh gradient (CE+KL) ...", flush=True)
    for m in mods:
        m.flip_grad.zero_()
    core.train()                                    # the dequant hook only fires in train mode
    for k, bi in enumerate(pr_idx):
        ids = batches[bi].to(dev)
        ti = cache["idx"][bi].unsqueeze(0).to(dev); tv = cache["val"][bi].unsqueeze(0).to(dev)
        hid.clear(); core(ids, logits_to_keep=1); h = hid["h"]
        loss = chunked_hidden_state_loss(h, Wlm, ti, tv, temperature=1.0, loss_type="cakld", chunk_size=4096)
        if args.ce_weight > 0:
            H = h[0, :-1, :]; tg = ids[0, 1:]
            n = args.ce_positions
            if 0 < n < H.shape[0]:
                sel = torch.arange(0, H.shape[0], max(1, H.shape[0] // n), device=H.device)[:n]
                H, tg = H[sel], tg[sel]
            loss = loss + args.ce_weight * chunked_ce(H, Wlm, tg)
        loss.backward()
        core.zero_grad(set_to_none=True)
        hid.clear(); del loss, h
        if (k + 1) % 4 == 0:
            print(f"    probe {k+1}/{len(pr_idx)}", flush=True)
    for m in mods:
        m.flip_grad /= max(1, len(pr_idx))
    core.eval()
    try:
        core.gradient_checkpointing_disable()        # sweep evals are no_grad; ckpt only slows them
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ── budget sweep: commit top-K by predicted gain (eta→∞ ⇒ pure first-order), measure, revert ──
    rows = []
    for frac in [float(x) for x in args.budgets.split(",")]:
        committed, revert = oracle_flip(mods, frac)
        kl, ce, ag = losses(ev_idx, want_agree=True)
        tot = kl + args.ce_weight * ce
        rec = 100.0 * (ag - b_ag) / max(1e-9, (100.0 - b_ag))    # % of FP gap recovered
        rows.append({"budget": frac, "committed": committed,
                     "KL": kl, "CE": ce, "CE+KL": tot,
                     "agreement": ag, "d_agreement": ag - b_ag, "pct_fp_gap": rec})
        print(f"  K={frac:<7.0e} flips={committed:>10,} ({100.0*committed/ntot:6.3f}%)  "
              f"CE+KL {tot:.4f} ({tot-b_tot:+.4f})  agree {ag:.2f}% ({ag-b_ag:+.2f})  "
              f"FP-gap recovered {rec:+.1f}%", flush=True)
        for (m, sel, old) in revert:
            m.tern_b[sel] = old                                   # restore before the next budget
        del revert
        torch.cuda.empty_cache()

    out = {"baseline": {"KL": b_kl, "CE": b_ce, "CE+KL": b_tot, "agreement": b_ag,
                        "fp_gap_points": 100.0 - b_ag},
           "scope": args.scope, "n_trits": ntot, "ce_weight": args.ce_weight, "sweep": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    best = max((r["pct_fp_gap"] for r in rows), default=0.0)
    print(f"\n  BEST FP-gap recovery across budgets: {best:+.1f}%")
    print("  >=50% ⇒ ASSIGNMENT-LIMITED (build the flip-control machinery; it's size-free)")
    print("  <=20% ⇒ SUGGESTIVE of capacity-limited — but this is first-order greedy with FIXED scales,")
    print("          a doubly-conservative lower bound. Escalate to AdaRound/OBQ-with-compensation before")
    print("          concluding the assignment lever is exhausted.")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
