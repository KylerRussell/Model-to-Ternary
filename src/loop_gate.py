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


class DRYLogitsProcessor(LogitsProcessor):
    """DRY (Don't Repeat Yourself) suffix-continuation penalty — arXiv:2608.22761, and the sampler
    shipped in llama.cpp / ExLlamaV2 / text-generation-webui.

    Unlike a flat repetition penalty (which penalises tokens uniformly wherever they occurred and
    wrecks code/LaTeX/structured text), DRY penalises ONLY the tokens that would EXTEND a repeat.
    For the current context s, find the longest suffix s[n-L:n] that also occurred earlier ending at
    j; the token s[j+1] that followed it is the one about to continue the loop, and it is penalised by

        multiplier * base ** (L - allowed_length)      for L >= allowed_length

    so the penalty grows exponentially with how much context is already repeating.

    Longest-suffix matching is the Z-algorithm on the reversed context: for reversed r, Z[i] is the
    longest common prefix of r and r[i:], which is exactly the longest common suffix of s[:n-i] and
    s[:n]. The continuation token is then s[n-i]. O(n) per step.

    WHY THIS ONE FIRST (13ag): PLAER = 0.400, so only ~40% of our loops hold an extractable answer.
    DRY is the only candidate whose value does NOT scale with PLAER -- it suppresses verbatim
    continuation whether or not an answer was ever derived.

    Deviation from llama.cpp, recorded: sequence breakers are applied by capping the match length at
    the distance to the most recent breaker token, rather than by llama.cpp's per-restart bookkeeping.
    Same intent (do not let a match run across a structural boundary), simpler implementation."""

    CAP = 1e4          # an absolute ban; keeps the exponential from overflowing on long loops

    def __init__(self, multiplier, base, allowed_length, breaker_ids, penalty_last_n=0):
        self.mult = float(multiplier); self.base = float(base)
        self.allowed = int(allowed_length); self.breakers = set(int(b) for b in breaker_ids)
        self.last_n = int(penalty_last_n)          # 0 = whole context

    @staticmethod
    def _z(r):
        n = len(r); z = [0] * n
        if n:
            z[0] = n
        l = rgt = 0
        for i in range(1, n):
            zi = 0
            if i < rgt:
                zi = min(rgt - i, z[i - l])
            while i + zi < n and r[zi] == r[i + zi]:
                zi += 1
            z[i] = zi
            if i + zi > rgt:
                l, rgt = i, i + zi
        return z

    def __call__(self, input_ids, scores):
        if self.mult <= 0:
            return scores
        for b in range(input_ids.shape[0]):
            s = input_ids[b].tolist()
            if self.last_n > 0:
                s = s[-self.last_n:]
            n = len(s)
            if n < self.allowed + 1:
                continue
            cap = n                                  # do not let a match cross a sequence breaker
            for k in range(n - 1, -1, -1):
                if s[k] in self.breakers:
                    cap = n - 1 - k
                    break
            if cap < self.allowed:
                continue
            z = self._z(s[::-1])
            pen = {}
            for i in range(1, n):
                L = min(z[i], cap)
                if L >= self.allowed:
                    tok = s[n - i]
                    e = L - self.allowed
                    # CLAMP. A real loop repeats for hundreds of tokens, so base**e overflows a
                    # Python float (1.75**1100 -> OverflowError, which killed the first run; the
                    # unit test had only gone to a 32-token repeat). Past ~1e4 the distinction is
                    # meaningless anyway: subtracting 1e4 from a logit is already an absolute ban
                    # on that token after softmax.
                    p = self.CAP if e > 64 else min(self.mult * (self.base ** e), self.CAP)
                    if p > pen.get(tok, 0.0):
                        pen[tok] = p
            for tok, p in pen.items():
                scores[b, tok] -= p
        return scores

from transformers import AutoModelForCausalLM, AutoTokenizer
from e2e_qp_distill import build_student, BLOCK_SIZE
from gen_reasoning_traces import PROMPTS as REASON_PROMPTS
try:
    from build_chat_calib import EASY, PLAIN
except Exception:
    EASY, PLAIN = [], []

SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
