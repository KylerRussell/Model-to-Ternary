#!/usr/bin/env python
"""loop_gate.py — the FREE-GEN gate that replaces the (proven-blind) teacher-forced probe (researcher round-4).
Greedy self-rollout in the HF harness, long enough to enter the 200-600-tok repetition attractor, scored with
the zlib compression-ratio repetition metric (arXiv:2604.08527) + truncation/commit rate. This is the go/no-go
signal for the exposure-bias residual — teacher-forced KL cannot see it.

  loop_rate  : frac of rollouts flagged repetitive (zlib comp-ratio > TAU  OR  a 5-gram/sentence repeated 3x)
  trunc_rate : frac that hit the token budget WITHOUT emitting EOS/</think>  (over-thinking / no-commit)
  commit_rate: frac (thinking-mode) that emitted </think> AND a non-empty, non-looping answer

Env: ORIG, E2E_MODEL, N_PREFIX, MAXNEW, TAU, THINK(1/0), TEMP, BATCH, SEED, GATE_OUT.  Model -> cuda:0.

REPRODUCIBILITY (added 2026-09-03). At TEMP>0 this gate samples, and it had NO seed, so every
invocation drew a different RNG stream. Measured on ONE unchanged model (4B e2eqp), three runs at
identical settings:

    pipeline N=48   loop 0.3125   commit 0.7917   comp 3.248
    rerun    N=48   loop 0.7083   commit 0.3542   comp 4.629
    rerun    N=96   loop 0.6875   commit 0.4062   comp 4.573

A 0.396 swing in loop_rate -- ~6 binomial SE, so NOT sampling noise of the reported kind. Looping is
bistable per prompt and the whole prompt set sits near that boundary, so one RNG stream flips many
prompts at once. Any single unseeded run is therefore uninterpretable, and 13v's "Gate B fails by one
sequence" conclusion came from the 0.3125 outlier. SEED now makes a run reproducible; report a MEAN
OVER SEEDS (and its spread) before comparing two models."""
import os, re, json, zlib, torch
from transformers import LogitsProcessor


from transformers import AutoModelForCausalLM, AutoTokenizer
from dry import DRYLogitsProcessor          # extracted; see dry.py


from e2e_qp_distill import build_student, BLOCK_SIZE
from gen_reasoning_traces import PROMPTS as REASON_PROMPTS
try:
    from build_chat_calib import EASY, PLAIN
except Exception:
    EASY, PLAIN = [], []

SEED = int(os.environ.get("SEED", "0"))     # seeding itself is deferred into the __main__ guard


ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
E2E = os.environ.get("E2E_MODEL", "output_4b/chatfix_v2/e2e_v1/modified_model")
N_PREFIX = int(os.environ.get("N_PREFIX", "48"))
MAXNEW = int(os.environ.get("MAXNEW", "768"))       # must exceed the 200-600-tok attractor onset
TAU = float(os.environ.get("TAU", "4.0"))           # zlib comp-ratio threshold (calibrate FP~2-3 vs loop>>)
THINK = os.environ.get("THINK", "1") == "1"
TEMP = float(os.environ.get("TEMP", "0.0"))         # 0 = greedy (matches deploy temp-0); can pass 0.6
BATCH = int(os.environ.get("BATCH", "8"))
THINK_CLOSE, EOS = 248069, 248046

tok = AutoTokenizer.from_pretrained(ORIG, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# round-robin interleave the three prompt types so any N covers easy / hard-reasoning / open-ended alike
# (open-ended like haiku/poems are where real loops appear; all-EASY would flatter the model).
_pools = [list(EASY), list(REASON_PROMPTS), list(PLAIN)]
_pools = [p for p in _pools if p] or [list(REASON_PROMPTS)]
prompts = []
i = 0
while len(prompts) < N_PREFIX:
    pool = _pools[i % len(_pools)]
    prompts.append(pool[(i // len(_pools)) % len(pool)])
    i += 1


def comp_ratio(text):
    b = text.encode("utf-8", "ignore")
    if len(b) < 40:
        return 1.0
    return len(b) / max(len(zlib.compress(b, 6)), 1)


def ngram_loop(text, n=5, rep=5):
    """HARD degeneration only: a 5-gram repeated >=5x, or a full sentence (>=5 words) repeated >=4x.
    (rep=3 over-flags legitimate reasoning that restates drafts — FP itself trips it; comp_ratio is primary.)"""
    w = text.split()
    seen = {}
    for i in range(len(w) - n):
        g = " ".join(w[i:i + n]); seen[g] = seen.get(g, 0) + 1
        if seen[g] >= rep:
            return True
    ss = {}
    for s in (x.strip() for x in re.split(r'[.\n!?]+', text)):
        if len(s.split()) >= 5:
            ss[s] = ss.get(s, 0) + 1
            if ss[s] >= 4:
                return True
    return False


if __name__ == "__main__":
    # Everything below is the GATE RUN. It is guarded because importing this module
    # used to execute it: a full 48-prompt generation sweep, discarded, on every
    # `from loop_gate import ...`. That was ~half of every GSM8K hour this project
    # spent, and it advanced the RNG stream by 48 sampled rollouts, so every recorded
    # result was really "seed N, plus a loop_gate run". Removing it CHANGES SAMPLED
    # OUTPUT and requires re-baselining -- see 13ap-iii.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"loop-gate: model={E2E} | N={N_PREFIX} maxnew={MAXNEW} think={THINK} temp={TEMP} tau={TAU} "
          f"seed={SEED}", flush=True)
    if os.environ.get("MODEL_KIND", "tern") == "fp":                # FP baseline for TAU calibration
        st = AutoModelForCausalLM.from_pretrained(E2E, trust_remote_code=True,
                                                  dtype=torch.bfloat16).to("cuda:0").eval()
        st.config.use_cache = True
    else:
        st, _ = build_student(E2E, ORIG, BLOCK_SIZE, "cuda:0"); st.eval(); st.config.use_cache = True
    # THINK_ROW_SCALE: multiply ALL per-block scales of the </think> row of the ternary lm_head by c>1. This is a
    # per-token logit gain that is fully ternary and foldable (assignments untouched, only the fp16/8-bit scales
    # change) — the researcher round-6 post-hoc calibration. It addresses the terminal-token CALIBRATION deficit
    # only; it cannot fix looping (a loop token at p~0.9 is not beaten by a globally-scaled close logit).
    if os.environ.get("HEAD_MODE"):                 # parity with math_correct/kl_flips_eval
        from head_swap import swap_head
        swap_head(st, os.environ.get("HEAD_SRC", "output_4bpipe/rotbase/modified_model"),
                  os.environ["HEAD_MODE"])
    if os.environ.get("EMBED_MODE"):
        from head_swap import swap_embed
        swap_embed(st, os.environ.get("HEAD_SRC", "output_4bpipe/rotbase/modified_model"),
                   os.environ["EMBED_MODE"])
    TRS = float(os.environ.get("THINK_ROW_SCALE", "1.0"))
    if TRS != 1.0:
        import torch.nn as _nn
        done_trs = False
        for _n, _m in st.named_modules():
            if _n.endswith("lm_head") and hasattr(_m, "scale") and hasattr(_m, "block_size"):
                bpr = _m.in_features // _m.block_size
                with torch.no_grad():
                    _m.scale[THINK_CLOSE * bpr:(THINK_CLOSE + 1) * bpr] *= TRS
                print(f"THINK_ROW_SCALE={TRS} -> scaled {bpr} block-scales of lm_head row {THINK_CLOSE}", flush=True)
                done_trs = True
            elif _n.endswith("lm_head") and isinstance(_m, _nn.Linear):
                with torch.no_grad():
                    _m.weight[THINK_CLOSE] *= TRS       # plain-Linear fallback (still on-grid: row x c)
                print(f"THINK_ROW_SCALE={TRS} -> scaled lm_head.weight row {THINK_CLOSE}", flush=True)
                done_trs = True
        if not done_trs:
            print("WARNING: THINK_ROW_SCALE set but no lm_head found", flush=True)

    if os.environ.get("DENSE_INFER", "0") == "1" and os.environ.get("MODEL_KIND", "tern") != "fp":
        from densify import densify, densify_selftest    # AFTER the TRS edit -- densify snapshots
        densify_selftest(st)                             # scales, so order matters
        densify(st)

    SAMPLES = os.environ.get("GATE_SAMPLES")                        # optional: dump generations for eyeballing
    sample_rows = []

    loops = truncs = commits = 0
    ratios = []
    think_lens = []
    # CRITICAL: <|im_end|>=248046 is the turn-end, but config.eos_token_id is None → generate never stops and
    # fills the budget by repeating <|im_end|> (a FAKE loop). Pass eos explicitly so generation stops correctly.
    gen_kw = dict(max_new_tokens=MAXNEW, pad_token_id=tok.eos_token_id, eos_token_id=EOS)
    # DRY sampler (13ag). OFF by default (DRY_MULT=0) so every previously recorded Gate B number stays
    # reproducible bit-for-bit. Defaults below are llama.cpp's when enabled.
    _DRY_MULT = float(os.environ.get("DRY_MULT", "0"))
    if _DRY_MULT > 0:
        from transformers import LogitsProcessorList
        _brk = []
        for _t in ("\n", ":", '"', "*", ".", ","):          # llama.cpp's default sequence breakers
            try:
                _ids = tok.encode(_t, add_special_tokens=False)
                if len(_ids) == 1:
                    _brk.append(_ids[0])
            except Exception:
                pass
        _dry = DRYLogitsProcessor(_DRY_MULT, float(os.environ.get("DRY_BASE", "1.75")),
                                  int(os.environ.get("DRY_ALLOWED", "2")), _brk,
                                  int(os.environ.get("DRY_LAST_N", "0")))
        gen_kw["logits_processor"] = LogitsProcessorList([_dry])
        print(f"DRY ON: mult={_DRY_MULT} base={os.environ.get('DRY_BASE','1.75')} "
              f"allowed={os.environ.get('DRY_ALLOWED','2')} breakers={len(_brk)}", flush=True)
    if TEMP > 0:
        gen_kw.update(do_sample=True, temperature=TEMP, top_p=0.95, top_k=20)
    else:
        gen_kw.update(do_sample=False)

    with torch.no_grad():
        for b0 in range(0, len(prompts), BATCH):
            chunk = prompts[b0:b0 + BATCH]
            msgs = [[{"role": "user", "content": p}] for p in chunk]
            try:
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=THINK,
                                              return_tensors="pt", tokenize=True, padding=True, return_dict=True)
            except TypeError:
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                              tokenize=True, padding=True, return_dict=True)
            ii = enc["input_ids"].to("cuda:0"); am = enc["attention_mask"].to("cuda:0")
            out = st.generate(ii, attention_mask=am, **gen_kw)
            for j in range(out.shape[0]):
                gen_ids = out[j, ii.shape[1]:].tolist()
                emitted_eos = EOS in gen_ids
                if emitted_eos:                          # truncate BATCH PADDING (pad_token==EOS==<|im_end|>)
                    gen_ids = gen_ids[: gen_ids.index(EOS) + 1]   # keep the real generation through the stop only
                emitted_close = THINK_CLOSE in gen_ids
                txt = tok.decode(gen_ids, skip_special_tokens=False)
                cr = comp_ratio(txt); ratios.append(cr)
                ng = ngram_loop(txt)
                looped = (cr > TAU) or ng
                loops += looped
                if SAMPLES is not None:
                    # FULL text, not txt[:800]. 800 chars is ~200 tokens and the answer, when it exists,
                    # lands at 300-600 tokens -- so the old cap truncated away exactly the evidence
                    # needed to tell COMMITMENT failure (answer present, model won't stop) from
                    # PATH-FINDING failure (no answer ever derived). Those need opposite fixes:
                    # decoding-time control vs better weights. Generation is ~2.3h; analysis of the
                    # dump is free, so save everything and analyse offline.
                    sample_rows.append({"prompt": chunk[j], "comp_ratio": round(cr, 2), "ngram": ng,
                                        "n_tok": len(gen_ids), "looped": bool(looped),
                                        "emitted_close": bool(emitted_close), "emitted_eos": bool(emitted_eos),
                                        "text": txt})
                # truncated / no-commit: hit budget with no stop token
                no_stop = not (emitted_eos or (THINK and emitted_close and
                               len(gen_ids) - gen_ids.index(THINK_CLOSE) > 2))
                truncs += no_stop
                if emitted_close:
                    think_lens.append(gen_ids.index(THINK_CLOSE))     # tokens of reasoning before the close
                if THINK:
                    answer = gen_ids[gen_ids.index(THINK_CLOSE)+1:] if emitted_close else []
                    commits += (emitted_close and len(answer) >= 2 and not ngram_loop(tok.decode(answer)))
                else:
                    commits += (emitted_eos and not looped)
            print(f"  {b0+len(chunk)}/{len(prompts)}  loop={loops} trunc={truncs} commit={commits}", flush=True)

    # ───────────────────────── SCORED SUBSET (13aq) ─────────────────────────
    # Gate B scored loop/commit/compression and NEVER checked whether the committed answer was
    # RIGHT. 13aj: c=1.40 passed every bar at 2.2% accuracy. 13ao: the baseline's only "correct"
    # answers were on problems the FP teacher itself fails. A commit_rate that does not require
    # correctness measures the rate of CONFIDENTLY-WRONG completions, and every commit gain this
    # project recorded -- OPSA's +0.083, DRY's, the row gain's +0.097 -- measured exactly that.
    #
    # These prompts run AFTER the main loop, in their own batches, with their own token budget, so
    # the loop/comp/trunc/commit numbers above are computed over exactly the same rollouts as before.
    N_SCORE = int(os.environ.get("N_SCORE", "24"))
    SCORE_MAXNEW = int(os.environ.get("SCORE_MAXNEW", "2048"))   # GSM8K needs room to close
    sc = {"n": 0, "correct": 0, "closed": 0, "commit": 0, "commit_correct": 0, "think_lens": []}
    if N_SCORE > 0:
        from datasets import load_dataset
        from answer_score import gold, pred, same
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N_SCORE))
        sgen = dict(gen_kw); sgen["max_new_tokens"] = SCORE_MAXNEW
        print(f"\n  scored subset: {N_SCORE} GSM8K problems, maxnew={SCORE_MAXNEW}", flush=True)
        for b0 in range(0, N_SCORE, BATCH):
            rows = [ds[j] for j in range(b0, min(b0 + BATCH, N_SCORE))]
            msgs = [[{"role": "user", "content": r["question"]}] for r in rows]
            try:
                e = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=THINK,
                                            return_tensors="pt", tokenize=True, padding=True,
                                            return_dict=True)
            except TypeError:
                e = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                            tokenize=True, padding=True, return_dict=True)
            ii2 = e["input_ids"].to("cuda:0")
            o = st.generate(ii2, attention_mask=e["attention_mask"].to("cuda:0"), **sgen)
            for j, r in enumerate(rows):
                g = o[j, ii2.shape[1]:].tolist()
                if EOS in g:                       # same padding trim as the main loop (13ap-i)
                    g = g[: g.index(EOS) + 1]
                txt = tok.decode(g, skip_special_tokens=False)
                closed = THINK_CLOSE in g
                ok = same(pred(txt), gold(r["answer"]))
                ans = g[g.index(THINK_CLOSE) + 1:] if closed else []
                committed = bool(closed and len(ans) >= 2 and not ngram_loop(tok.decode(ans)))
                sc["n"] += 1; sc["correct"] += ok; sc["closed"] += closed
                sc["commit"] += committed; sc["commit_correct"] += (committed and ok)
                if closed:
                    sc["think_lens"].append(g.index(THINK_CLOSE))
            print(f"  scored {sc['n']}/{N_SCORE}  correct={sc['correct']} "
                  f"commit_correct={sc['commit_correct']}", flush=True)

    n = len(prompts)
    mean_cr = sum(ratios) / n
    res = {"model": E2E, "n": n, "think": THINK, "temp": TEMP, "maxnew": MAXNEW, "tau": TAU,
           "seed": SEED,
        "loop_rate": loops / n, "trunc_rate": truncs / n, "commit_rate": commits / n,
           "mean_comp_ratio": mean_cr, "max_comp_ratio": max(ratios),
           "mean_think_len": (sum(think_lens)/len(think_lens)) if think_lens else None,
           "n_closed": len(think_lens)}
    print("\n" + "=" * 60)
    print(f"  loop_rate   {100*loops/n:5.1f}%   ({loops}/{n})")
    print(f"  trunc_rate  {100*truncs/n:5.1f}%   (no EOS/</think> at budget)")
    print(f"  commit_rate {100*commits/n:5.1f}%")
    print(f"  comp_ratio  mean {mean_cr:.2f}  max {max(ratios):.2f}  (FP-like ~2-3, loopy >>{TAU})")
    if think_lens:
        print(f"  think_len   mean {sum(think_lens)/len(think_lens):.0f} tok over {len(think_lens)} closers "
              f"(watch for COLLAPSE = premature closing)")
    print("=" * 60)
    # ── scored metrics + VERDICT ─────────────────────────────────────────────
    sn = sc["n"]
    res["scored_n"] = sn
    stl = (sum(sc["think_lens"]) / len(sc["think_lens"])) if sc["think_lens"] else None
    if sn:
        res.update(score_acc=sc["correct"] / sn, score_closed_rate=sc["closed"] / sn,
                   score_acc_given_closed=sc["correct"] / max(sc["closed"], 1),
                   score_commit_rate=sc["commit"] / sn,
                   commit_correct_rate=sc["commit_correct"] / sn, score_think_len=stl)
        print(f"  SCORED SUBSET ({sn} GSM8K problems)")
        print(f"    accuracy            {100*sc['correct']/sn:5.1f}%   ({sc['correct']}/{sn})")
        print(f"    closed_rate         {100*sc['closed']/sn:5.1f}%")
        print(f"    commit_rate         {100*sc['commit']/sn:5.1f}%   (answer emitted, not looping)")
        print(f"    commit_CORRECT_rate {100*sc['commit_correct']/sn:5.1f}%   "
              f"<- committed AND right; the gate bar")
        if stl is not None:
            print(f"    think_len           {stl:.0f} tok")
        print("=" * 60)

    LOOP_BAR = float(os.environ.get("LOOP_BAR", "0.30"))
    COMMIT_BAR = float(os.environ.get("COMMIT_BAR", "0.68"))
    COMP_BAR = float(os.environ.get("COMP_BAR", "3.1"))
    CORRECT_BAR = float(os.environ.get("CORRECT_BAR", "0.20"))
    TEACHER_TL = float(os.environ.get("TEACHER_THINK_LEN", "0") or 0)
    THINK_FLOOR = float(os.environ.get("THINK_FLOOR", "0.75"))
    checks = [("loop_rate", loops / n, "<=", LOOP_BAR),
              ("commit_rate", commits / n, ">=", COMMIT_BAR),
              ("comp_ratio", mean_cr, "<=", COMP_BAR)]
    if sn:
        checks.append(("commit_correct_rate", sc["commit_correct"] / sn, ">=", CORRECT_BAR))
        # think_len guard (13aj): c=1.40 passed every behavioural bar while think_len COLLAPSED to
        # 224 against the teacher's 586-648 -- it had stopped reasoning. A behavioural gate cannot
        # see that; only a length comparison against the teacher can.
        if TEACHER_TL > 0 and stl is not None:
            checks.append(("think_len/teacher", stl / TEACHER_TL, ">=", THINK_FLOOR))
    ok_all = all((v <= b) if op == "<=" else (v >= b) for _, v, op, b in checks)
    for name, v, op, b in checks:
        good = (v <= b) if op == "<=" else (v >= b)
        print(f"  {'PASS' if good else 'FAIL'}  {name:20s} {v:8.4f} {op} {b}")
    if not sn:
        # The gate MUST NOT be able to report PASS without a correctness measurement. That is
        # exactly how c=1.40 passed every bar at 2.2% accuracy (13aj/13am).
        res["verdict"] = "UNSCORED"
        print("  VERDICT: UNSCORED — N_SCORE=0, so correctness was never checked. This is NOT a pass.")
    else:
        res["verdict"] = "PASS" if ok_all else "FAIL"
        print(f"  VERDICT: {res['verdict']}")
    print("=" * 60)
    if os.environ.get("GATE_OUT"):
        res["dry_mult"] = _DRY_MULT
        if _DRY_MULT > 0:
            res["dry_base"] = float(os.environ.get("DRY_BASE", "1.75"))
            res["dry_allowed"] = int(os.environ.get("DRY_ALLOWED", "2"))
        json.dump(res, open(os.environ["GATE_OUT"], "w"), indent=1)
        print("wrote", os.environ["GATE_OUT"])
    if SAMPLES is not None:
        json.dump(sample_rows, open(SAMPLES, "w"), indent=1)
        print("wrote samples", SAMPLES)
