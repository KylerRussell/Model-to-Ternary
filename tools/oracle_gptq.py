#!/usr/bin/env python
"""oracle_gptq.py — SECOND-ORDER oracle for the ASSIGNMENT lever (follow-up to §8o).

§8o showed FIRST-ORDER flip ranking (Δ̂=ḡ·s·(t'−t)) is invalid: a flip moves a weight by s·d ≈ its own
magnitude, so the linear term is meaningless and every committed budget made the model WORSE. That measured
the SEARCH, not the lever. This re-asks the question with the method that actually handles a large discrete
step: GPTQ/OBC — Hessian-weighted rounding WITH error compensation through H⁻¹.

TEST: re-solve down_proj's ternary assignment at the CURRENT operating point.
  1. Collect H = XᵀX at each down_proj INPUT from the CURRENT student (post-E2E scales, quantized-path
     activations) — i.e. the operating point has drifted since block-AP placed these trits.
  2. Re-run _gptq_ternary(W_fp, H, g64) from the ROTATED FP weights (the same targets block-AP used).
  3. Install and measure held-out CE+KL + FP-agreement vs the current model.

READING:
  BETTER  → a second-order re-solve at the current state finds a better assignment ⇒ assignment headroom
            exists, and the report's V/P alternation (refit assignments after scales move) is validated.
  SAME/WORSE → the existing GPTQ assignment is already near-optimal for this objective at this granularity;
            combined with §8o that is real evidence the assignment lever is close to exhausted (NOT proof —
            a loss-aware AdaRound search could still differ).
CONFOUND (reported, not hidden): the fresh GPTQ brings its own absmax scales, discarding E2E's trained scales.
So "worse" may reflect the lost scale training rather than worse assignments. --keep-scales re-fits the OLD
trained scale onto the NEW trits to separate the two.

  ORIG_MODEL=<orig-config> TERNARY_BLOCK_SIZE=64 ./.venv/bin/python tools/oracle_gptq.py \
      --student output_4bpipe/e2eqp/modified_model --fp output_4bpipe/rotbase/modified_model \
      --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt
"""
import argparse, os, sys, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from e2e_qp_distill import (build_student, load_calib_batches, BLOCK_SIZE, TernaryScaleLinear,
                            chunked_hidden_state_loss, chunked_ce, _chunked_argmax,
                            extract_ternary_scale, pack_2bit)
from block_ap_recovery import _gptq_ternary
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--fp", required=True, help="rotated FP model (the GPTQ targets)")
    ap.add_argument("--orig-config-path", default=os.environ.get("ORIG_MODEL"))
    ap.add_argument("--calib", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--hess-n", type=int, default=24, help="#seqs for the Hessian")
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--ce-weight", type=float, default=0.5)
    ap.add_argument("--ce-positions", type=int, default=512)
    ap.add_argument("--seq", type=int, default=2560)
    ap.add_argument("--keep-scales", action="store_true",
                    help="re-fit the OLD trained scale onto the NEW trits (isolates the assignment change)")
    ap.add_argument("--out", default="logs/oracle_gptq.json")
    args = ap.parse_args()
    dev = "cuda:0"

    print(f"loading student {args.student} (block {BLOCK_SIZE}) ...", flush=True)
    core, _ = build_student(args.student, args.orig_config_path, BLOCK_SIZE, dev)
    core.config.use_cache = False
    for p in core.parameters():
        p.requires_grad_(False)
    core.eval()

    targets = [(n, m) for n, m in core.named_modules()
               if isinstance(m, TernaryScaleLinear) and n.endswith("down_proj")]
    print(f"  {len(targets)} down_proj modules", flush=True)

    base_mod = core.model.language_model if hasattr(core.model, "language_model") else core.model
    hid = {}
    base_mod.norm.register_forward_hook(lambda mod, i, o: hid.__setitem__("h", o))
    with torch.no_grad():
        w = core.lm_head.dequant() if hasattr(core.lm_head, "dequant") else core.lm_head.weight
        Wlm = w.detach().to(torch.bfloat16).cpu().contiguous(); del w
    V = Wlm.shape[0]
    core.lm_head.forward = (lambda x, _v=V: x.new_zeros(x.shape[:-1] + (_v,)))
    torch.cuda.empty_cache()

    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")
    cache = torch.load(args.teacher_cache, map_location="cpu")
    n_avail = min(len(batches), len(cache["idx"]))
    ev_idx = list(range(n_avail - args.eval_n, n_avail))
    hs_idx = list(range(n_avail - args.eval_n - args.hess_n, n_avail - args.eval_n))

    def losses(idx_list):
        tk = ta = 0.0; agree = 0; ntok = 0
        with torch.no_grad():
            for bi in idx_list:
                ids = batches[bi].to(dev)
                ti = cache["idx"][bi].unsqueeze(0).to(dev); tv = cache["val"][bi].unsqueeze(0).to(dev)
                hid.clear(); core(ids, logits_to_keep=1); h = hid["h"]
                tk += float(chunked_hidden_state_loss(h, Wlm, ti, tv, temperature=1.0,
                                                      loss_type="cakld", chunk_size=4096))
                if args.ce_weight > 0:
                    H_ = h[0, :-1, :]; tg = ids[0, 1:]
                    n = args.ce_positions
                    if 0 < n < H_.shape[0]:
                        sel = torch.arange(0, H_.shape[0], max(1, H_.shape[0] // n), device=H_.device)[:n]
                        H_, tg = H_[sel], tg[sel]
                    ta += float(chunked_ce(H_, Wlm, tg))
                am = _chunked_argmax(h[0, :-1, :], Wlm, chunk=4096)
                agree += int((am == ti[0, :-1, 0]).sum()); ntok += h.shape[1] - 1
        n = max(1, len(idx_list))
        return tk / n, ta / n, 100.0 * agree / max(1, ntok)

    b_kl, b_ce, b_ag = losses(ev_idx)
    b_tot = b_kl + args.ce_weight * b_ce
    print(f"\n  BASELINE: KL {b_kl:.4f} | CE {b_ce:.4f} | CE+KL {b_tot:.4f} | agreement {b_ag:.2f}%")
    print(f"  FP gap = {100.0 - b_ag:.2f} points\n", flush=True)

    # ── collect H = XᵀX at each down_proj input, at the CURRENT operating point ──
    print(f"  collecting Hessians over {len(hs_idx)} seqs ...", flush=True)
    Hs = {}
    handles = []
    def mk(nme, mod):
        def hook(m, inp):
            x = (inp[0] if isinstance(inp, tuple) else inp).detach()
            x = x.reshape(-1, x.shape[-1]).float()
            if nme not in Hs:
                Hs[nme] = torch.zeros(x.shape[-1], x.shape[-1], device=dev, dtype=torch.float32)
            Hs[nme] += x.t() @ x
        return hook
    for n, m in targets:
        handles.append(m.register_forward_pre_hook(mk(n, m)))
    with torch.no_grad():
        for k, bi in enumerate(hs_idx):
            core(batches[bi].to(dev), logits_to_keep=1)
            hid.clear()
            if (k + 1) % 8 == 0:
                print(f"    hess {k+1}/{len(hs_idx)}  ({torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)
    for h_ in handles:
        h_.remove()
    print(f"  Hessians: {len(Hs)} × {tuple(next(iter(Hs.values())).shape)} "
          f"({sum(v.numel()*4 for v in Hs.values())/1e9:.1f}GB)", flush=True)

    # ── re-solve each down_proj with GPTQ (error-compensated) from the rotated FP weights ──
    fpf = safe_open(str(Path(args.fp) / "model.safetensors"), framework="pt")
    fp_names = {k for k in fpf.keys()}
    changed = 0; n_flip_tot = 0; n_tot = 0
    for n, m in targets:
        wname = n + ".weight"
        if wname not in fp_names:                      # student names may be prefixed differently
            # Same ambiguity fixed in e2e_qp_distill._fp_weight_lookup (2026-08-19): the suffix
            # `layers.0.mlp.down_proj.weight` matches BOTH the language-model tower and the MTP head,
            # and picking cand[0] out of a SET is order-dependent under string hash randomisation —
            # ~50% of runs silently grabbed the MTP head's weights for layer 0. Sort for determinism
            # and prefer the main `model.` tower.
            cand = sorted(k for k in fp_names if k.endswith(n.split("model.")[-1] + ".weight"))
            if not cand:
                print(f"    SKIP {n}: no FP weight found"); continue
            main = [k for k in cand if k.startswith("model.")]
            if main:
                cand = main
            if len(cand) > 1:
                print(f"    AMBIGUOUS {n}: {cand} -> using {cand[0]}")
            wname = cand[0]
        W_fp = fpf.get_tensor(wname).to(dev).float()
        H = Hs[n]
        old_deq = m.dequant().detach().clone()
        Q = _gptq_ternary(W_fp, H, BLOCK_SIZE)         # [out,inp] ternary×scale, error-compensated
        tern, scale = extract_ternary_scale(Q.float(), BLOCK_SIZE)
        if args.keep_scales:
            scale = m.scale.data.clone()               # keep E2E-trained scales, take only the new trits
        m.packed.data = pack_2bit(tern).to(dev)
        m.scale.data = scale.to(dev)
        # how many assignments actually changed
        o_t, _ = extract_ternary_scale(old_deq.float(), BLOCK_SIZE)
        n_flip_tot += int((o_t != tern).sum()); n_tot += tern.numel()
        changed += 1
        del W_fp, Q, old_deq, o_t
        Hs[n] = None
        torch.cuda.empty_cache()
    print(f"  re-solved {changed} layers; assignments changed {100.0*n_flip_tot/max(1,n_tot):.2f}% "
          f"({n_flip_tot:,}/{n_tot:,})", flush=True)

    kl, ce, ag = losses(ev_idx)
    tot = kl + args.ce_weight * ce
    rec = 100.0 * (ag - b_ag) / max(1e-9, (100.0 - b_ag))
    print(f"\n  GPTQ RE-SOLVE: KL {kl:.4f} | CE {ce:.4f} | CE+KL {tot:.4f} ({tot-b_tot:+.4f}) | "
          f"agreement {ag:.2f}% ({ag-b_ag:+.2f}) | FP-gap recovered {rec:+.1f}%")
    print("  BETTER ⇒ assignment headroom exists (V/P alternation validated)")
    print("  SAME/WORSE ⇒ existing GPTQ assignment already near-optimal at this granularity")
    if not args.keep_scales:
        print("  NOTE: fresh GPTQ uses its own absmax scales (E2E scale training discarded) — rerun with")
        print("        --keep-scales to isolate the assignment change from the lost scale training.")

    out = {"baseline": {"KL": b_kl, "CE": b_ce, "CE+KL": b_tot, "agreement": b_ag},
           "gptq": {"KL": kl, "CE": ce, "CE+KL": tot, "agreement": ag, "pct_fp_gap": rec},
           "assignments_changed_pct": 100.0*n_flip_tot/max(1, n_tot),
           "keep_scales": bool(args.keep_scales), "block_size": BLOCK_SIZE}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
