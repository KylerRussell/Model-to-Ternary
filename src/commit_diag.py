#!/usr/bin/env python
"""commit_diag.py — WHY does the ternary model over-think / loop instead of committing? At the exact positions
where the FP teacher emits a COMMIT token (</think> to end reasoning, or EOS to end the turn), measure whether
the ternary model ALSO wants to commit there (teacher-forced on the FP trace). Disambiguates the fix:

  * ternary UNDER-RANKS </think>/EOS at commit positions (even teacher-forced)  => SUPERVISION problem
    -> the model never learned to emit the stop token; fix = upweight/oversample commit positions in training.
  * ternary RANKS </think>/EOS #1 at commit positions (agrees with FP)          => EXPOSURE BIAS
    -> it can commit given the right context but free-gen drifts away; fix = stronger on-policy / anti-drift.

Env: ORIG, FP_DIR, E2E_MODEL, TRACES. FP->cuda:0, tern->cuda:1."""
import os, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from e2e_qp_distill import build_student, BLOCK_SIZE

ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
FP_DIR = os.environ.get("FP_DIR", "output_4b/rot/modified_model")
E2E = os.environ.get("E2E_MODEL", "output_4b/chatfix_v2/e2e_v1/modified_model")
TRACES = os.environ.get("TRACES", "output_4b/reasoning_traces.json")
THINK_CLOSE, EOS = 248069, 248046
SEQ_CAP = int(os.environ.get("SEQ_CAP", "2048"))

traces = json.load(open(TRACES))["traces"]
fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()
fp.config.use_cache = False
st, _ = build_student(E2E, ORIG, BLOCK_SIZE, "cuda:1"); st.eval(); st.config.use_cache = False
print(f"commit-diag: {len(traces)} traces | tern={E2E}", flush=True)


def stats(logits, target):
    """rank (0=top) and prob of target token in one position's logits."""
    lg = logits.float()
    prob = F.softmax(lg, -1)[target].item()
    rank = int((lg > lg[target]).sum().item())
    return rank, prob, int(lg.argmax())


acc = {"think_close": [], "eos": [], "content": []}   # each entry: (fp_rank, fp_prob, t_rank, t_prob, t_argmax, target)
with torch.no_grad():
    for t in traces:
        ids = t["ids"][:SEQ_CAP]
        T = len(ids)
        if T < 8:
            continue
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        lf = fp(x.to("cuda:0")).logits[0].cpu()
        lt = st(x.to("cuda:1")).logits[0].cpu()
        for pos in range(T - 1):
            tgt = ids[pos + 1]
            if tgt == THINK_CLOSE:
                key = "think_close"
            elif tgt == EOS:
                key = "eos"
            elif pos % 37 == 0:                 # sparse content-position control
                key = "content"
            else:
                continue
            fr, fpp, _ = stats(lf[pos], tgt)
            tr, tpp, ta = stats(lt[pos], tgt)
            acc[key].append((fr, fpp, tr, tpp, ta, tgt))
        del lf, lt
        torch.cuda.empty_cache()

print("\n" + "=" * 78)
print(f"{'position type':14s} {'n':>4s} {'FPrank':>7s} {'FPprob':>7s} {'TERNrank':>9s} {'TERNprob':>9s} {'tern_wants_it%':>14s}")
for key in ["think_close", "eos", "content"]:
    rows = acc[key]
    if not rows:
        print(f"{key:14s}   (none)")
        continue
    n = len(rows)
    fr = sum(r[0] for r in rows) / n
    fpp = sum(r[1] for r in rows) / n
    tr = sum(r[2] for r in rows) / n
    tpp = sum(r[3] for r in rows) / n
    wants = 100 * sum(1 for r in rows if r[4] == r[5]) / n     # tern argmax == the commit target
    print(f"{key:14s} {n:>4d} {fr:>7.1f} {fpp:>7.3f} {tr:>9.1f} {tpp:>9.3f} {wants:>13.1f}%")
print("=" * 78)
print("read: TERNrank>>FPrank & low tern_wants_it% at think_close/eos = SUPERVISION problem (under-emits stop);")
print("      TERNrank~FPrank & high tern_wants_it% = model CAN commit teacher-forced => EXPOSURE BIAS in free-gen.")
