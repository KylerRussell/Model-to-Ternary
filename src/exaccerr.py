#!/usr/bin/env python
"""exaccerr.py — does divergence ACCUMULATE along a generated chain, and does that predict whether
the chain got the right answer?

WHY THIS AND NOT ANOTHER METRIC. Every fidelity number this project has ever trusted -- eval2k top-1
agreement, mean KL, block-MSE -- is measured TEACHER-FORCED on the teacher's own tokens. Free
generation is not on that trajectory. The moment the student emits a token the teacher would not
have, every later token is conditioned on a prefix the 70%-agreement figure was never measured over.
That is exposure bias (Arora et al., arXiv:2204.01171), which proved perplexity tracks per-step error
at rho=0.9997 but length-normalized regret at only rho=0.4003 -- i.e. our metrics are structurally
incapable of seeing this, which is exactly what we observe (Gate A 78.50% PASS, GSM8K acc|closed
0.0714).

WHAT IS COMPUTED. One model pair (FP teacher vs ternary student), teacher-forced over a token
sequence that some model GENERATED. Per generated position t:
    d_t      = KL(p_FP || p_student)  at a state the generator actually visited
    flip_t   = argmax p_FP != argmax p_student          (the two models part company here)
    conf_t   = flip_t AND max p_FP > 0.5                (FP is sure -- a consequential divergence)
Changing only the TOKEN SOURCE turns this into the exposure-bias term:
    student-generated tokens -> d_t at student-visited states  (regret R)
    teacher-generated tokens -> d_t at teacher-visited states  (oracle error epsilon)
    %ExAccErr = 100 * (mean_d_student - mean_d_teacher) / mean_d_teacher

WHY PER-ROLLOUT AND NOT PER-CHECKPOINT. Ranking two checkpoints is n=2; a coin passes it half the
time. Every rollout here carries its own `ok` label from math_correct.py, so the metric is validated
WITHIN one checkpoint -- identical weights, identical seed, no cross-model confound -- by asking
whether it separates the rollouts that got the answer right from the ones that did not.

FALSIFIERS, stated in advance (this project's loss curves have caught zero failures, invariants all):
  * d_t must GROW with t on student traces and stay FLAT on teacher traces. If it grows on both, the
    growth is a property of long contexts, not of being off-distribution, and the mechanism is wrong.
  * the metric must separate the known-collapsed c=1.40 arm (1/46) from c=1.30 (2/28). If it cannot,
    it is not measuring chain survival.

Env: ROWS (token source, a *_rows.json from math_correct), FP_DIR, E2E_MODEL, ORIG, THINK_ROW_SCALE,
     NBIN, CHUNK, MAXROW, OUT.
"""
import os, json, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from e2e_qp_distill import build_student, BLOCK_SIZE

ROWS = os.environ["ROWS"]
ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
FP_DIR = os.environ.get("FP_DIR", "output_4bpipe/rotbase/modified_model")
E2E = os.environ["E2E_MODEL"]
TRS = float(os.environ.get("THINK_ROW_SCALE", "1.0"))
NBIN = int(os.environ.get("NBIN", "10"))          # position deciles for the growth profile
CHUNK = int(os.environ.get("CHUNK", "256"))       # positions per KL slice (vocab is 248320)
MAXROW = int(os.environ.get("MAXROW", "0"))
THINK_CLOSE = 248069

two_gpu = torch.cuda.device_count() >= 2
FP_DEV, T_DEV = ("cuda:0", "cuda:1") if two_gpu else ("cuda:0", "cuda:0")

rows = json.load(open(ROWS))
if MAXROW:
    rows = rows[:MAXROW]
assert "gen_ids" in rows[0], f"{ROWS} predates the trace patch — re-run math_correct.py"
print(f"ExAccErr: {len(rows)} rollouts from {ROWS}\n  FP={FP_DIR} ({FP_DEV})  tern={E2E} ({T_DEV}) "
      f"trs={TRS}", flush=True)

print("loading FP...", flush=True)
fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True,
                                          dtype=torch.bfloat16).to(FP_DEV).eval()
fp.config.use_cache = False
print("loading ternary...", flush=True)
st, _ = build_student(E2E, ORIG, BLOCK_SIZE, T_DEV)
st.eval()
st.config.use_cache = False

if TRS != 1.0:            # the rollouts were generated with this applied; the analysed student must
    for _n, _m in st.named_modules():          # be the same model that produced them
        if _n.endswith("lm_head") and hasattr(_m, "scale") and hasattr(_m, "block_size"):
            bpr = _m.in_features // _m.block_size
            with torch.no_grad():
                _m.scale[THINK_CLOSE * bpr:(THINK_CLOSE + 1) * bpr] *= TRS
            print(f"THINK_ROW_SCALE={TRS} applied", flush=True)
            break


if os.environ.get("DENSE_INFER", "0") == "1":
    from densify import densify, densify_selftest       # AFTER the TRS edit -- see densify.py
    densify_selftest(st)
    densify(st)


@torch.no_grad()
def analyse(prompt_ids, gen_ids):
    """Teacher-force BOTH models over prompt+gen and reduce to per-position scalars."""
    P, G = len(prompt_ids), len(gen_ids)
    full = torch.tensor([prompt_ids + gen_ids])
    lf = fp(full.to(FP_DEV)).logits[0]                     # [L, V] bf16
    lt = st(full.to(T_DEV)).logits[0]
    lo, hi = P - 1, P + G - 1                              # positions predicting gen[0..G-1]
    d, flip, conf, agree, ent = [], [], [], [], []
    for a in range(lo, hi, CHUNK):
        b = min(a + CHUNK, hi)
        cf = lf[a:b].float().to(T_DEV)
        ct = lt[a:b].float()
        pf = F.softmax(cf, -1)
        lpf = F.log_softmax(cf, -1)
        kl = (pf * (lpf - F.log_softmax(ct, -1))).sum(-1)
        # The FP teacher's own next-token entropy at the same position. The rival explanation for
        # any KL profile is that KL simply tracks local uncertainty (the overthinking-marker paper
        # reports rho=0.92 between high-KL positions and high next-token entropy). If d_t is a
        # function of H_t, a KL profile says nothing about accumulation and everything about where
        # the hard choices sit in a chain -- so measure H_t here rather than argue about it later.
        ent.append((-(pf * lpf).sum(-1)).cpu())
        af, at = cf.argmax(-1), ct.argmax(-1)
        d.append(kl.cpu())
        fl = (af != at)
        flip.append(fl.cpu())
        conf.append((fl & (pf.max(-1).values > 0.5)).cpu())
        # did the teacher endorse the token the generator actually emitted at this state?
        emitted = torch.tensor(gen_ids[a - lo:b - lo], device=T_DEV)
        agree.append((af == emitted).cpu())
        del cf, ct, pf, lpf, kl
    del lf, lt
    return (torch.cat(d), torch.cat(flip), torch.cat(conf), torch.cat(agree), torch.cat(ent))


def first_true(t):
    nz = torch.nonzero(t)
    return int(nz[0]) if len(nz) else -1


out = []
prof_sum, prof_n = [0.0] * NBIN, [0] * NBIN                # over the whole 2048-token budget
tprof_sum, tprof_n = [0.0] * NBIN, [0] * NBIN             # over the REASONING SPAN only
eprof_sum = [0.0] * NBIN
dh_x, dh_y = [], []                                        # pooled (d_t, H_t) for the correlation


def _accum(vals, ent_vals, ssum, snum, esum=None):
    n = len(vals)
    ix = (torch.arange(n, dtype=torch.float32) * NBIN / n).long().clamp(max=NBIN - 1)
    for b in range(NBIN):
        m = ix == b
        if int(m.sum()):
            ssum[b] += float(vals[m].sum())
            snum[b] += int(m.sum())
            if esum is not None:
                esum[b] += float(ent_vals[m].sum())


for i, r in enumerate(rows):
    g = r["gen_ids"]
    if len(g) < 8:
        continue
    d, flip, conf, agree, ent = analyse(r["prompt_ids"], g)
    G = len(d)
    idx = (torch.arange(G, dtype=torch.float32) * NBIN / G).long().clamp(max=NBIN - 1)
    _accum(d, ent, prof_sum, prof_n)
    # EVERY rollout here runs to the full MAXNEW budget, but the chain ends at </think> (teacher
    # ~970, student ~597). Positions past the close tag are trailing/looping text, which is highly
    # repetitive and therefore EASY to predict -- averaging it into the late deciles pushes KL down
    # for reasons that have nothing to do with the reasoning chain. Accumulation is a claim about
    # the chain, so it has to be measured over the chain.
    close = g.index(THINK_CLOSE) if THINK_CLOSE in g else G
    if close >= NBIN:
        _accum(d[:close], ent[:close], tprof_sum, tprof_n, eprof_sum)
    step = max(1, G // 256)                                # subsample for the pooled correlation
    dh_x.extend(d[::step].tolist())
    dh_y.extend(ent[::step].tolist())
    rec = {"ok": bool(r["ok"]), "closed": bool(r["closed"]), "n_tok": G, "think_len": close,
           "kl_think_span": float(d[:close].mean()),
           "mean_kl": float(d.mean()),
           "kl_first_decile": float(d[idx == 0].mean()),
           "kl_last_decile": float(d[idx == NBIN - 1].mean()),
           "flip_rate": float(flip.float().mean()),
           "conf_flip_rate": float(conf.float().mean()),
           "teacher_endorse_rate": float(agree.float().mean()),
           "first_flip": first_true(flip), "first_conf_flip": first_true(conf)}
    out.append(rec)
    if (i + 1) % 8 == 0 or i + 1 == len(rows):
        print(f"  {i+1}/{len(rows)}  mean_kl={sum(o['mean_kl'] for o in out)/len(out):.4f}",
              flush=True)

prof = [prof_sum[b] / max(prof_n[b], 1) for b in range(NBIN)]
tprof = [tprof_sum[b] / max(tprof_n[b], 1) for b in range(NBIN)]
eprof = [eprof_sum[b] / max(tprof_n[b], 1) for b in range(NBIN)]
xs, ys = torch.tensor(dh_x), torch.tensor(dh_y)
xs = xs - xs.mean(); ys = ys - ys.mean()
rho = float((xs * ys).sum() / (xs.norm() * ys.norm()).clamp_min(1e-9))
agg = {"rows": ROWS, "model": E2E, "fp": FP_DIR, "trs": TRS, "n": len(out),
       "mean_kl": sum(o["mean_kl"] for o in out) / max(len(out), 1),
       "mean_kl_think": sum(o["kl_think_span"] for o in out) / max(len(out), 1),
       "profile_full": prof, "profile_think": tprof, "profile_entropy_think": eprof,
       "pearson_kl_entropy": rho, "per_rollout": out}
print("\n" + "=" * 68)
print(f"  mean KL(fp||tern) at generator-visited states : {agg['mean_kl']:.4f} nats")
print(f"    restricted to the reasoning span (pre-close) : {agg['mean_kl_think']:.4f} nats")
print(f"  teacher endorses the emitted token            : "
      f"{100*sum(o['teacher_endorse_rate'] for o in out)/max(len(out),1):.2f}%")
print(f"  KL profile, FULL 2048-token budget:")
print("    " + "  ".join(f"{p:.3f}" for p in prof))
print(f"    first->last decile ratio: {prof[-1]/max(prof[0],1e-9):.3f}x")
print(f"  KL profile, REASONING SPAN only  <- the accumulation test:")
print("    " + "  ".join(f"{p:.3f}" for p in tprof))
print(f"    first->last decile ratio: {tprof[-1]/max(tprof[0],1e-9):.3f}x")
print(f"  FP next-token ENTROPY over the same span (nats):")
print("    " + "  ".join(f"{p:.3f}" for p in eprof))
print(f"    first->last decile ratio: {eprof[-1]/max(eprof[0],1e-9):.3f}x")
print(f"  pearson(d_t, H_t) pooled over positions       : {rho:+.4f}"
      f"   <- if this is high, the KL profile IS an entropy profile")
ok = [o for o in out if o["ok"]]
bad = [o for o in out if o["closed"] and not o["ok"]]
if ok and bad:
    for k in ("mean_kl", "conf_flip_rate", "first_conf_flip", "teacher_endorse_rate"):
        mo = sum(o[k] for o in ok) / len(ok)
        mb = sum(o[k] for o in bad) / len(bad)
        print(f"  {k:22s} correct(n={len(ok)}) {mo:9.4f}   wrong|closed(n={len(bad)}) {mb:9.4f}")
print("=" * 68)
if os.environ.get("OUT"):
    json.dump(agg, open(os.environ["OUT"], "w"), indent=1)
    print(f"-> {os.environ['OUT']}")
