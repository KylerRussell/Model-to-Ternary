#!/usr/bin/env python
"""math_correct.py — GSM8K accuracy under the SAME decoding config as the Gate B harness.

WHY. Gate B scores commit / loop / compression and **does not score correctness at all**. A model
that closes early with a WRONG answer records a clean "commit". 13aj showed how dangerous that is:
`THINK_ROW_SCALE` c=1.40 passed every Gate B bar while think_len collapsed to 224 vs the teacher's
586-648 — the model had stopped reasoning. think_len caught it, but think_len is only a PROXY.

The judgement now in front of us needs the real thing: c=1.25 keeps think_len inside the teacher's
band (617) but leaves commit 1.3 pp short of the bar, while c=1.30 passes the bar (0.7708) by
thinking 16% less than the teacher (492). Is that shortening free, or is it accuracy being traded
away? Only a scored answer settles it.

This reuses loop_gate's model construction and sampler wiring on purpose, so the numbers are
comparable to the gate: same build_student path (or MODEL_KIND=fp), same DRY processor, same
THINK_ROW_SCALE, same chat template / thinking mode / temperature / seed.

Env: ORIG, E2E_MODEL, MODEL_KIND, N_PROB, MAXNEW, THINK, TEMP, BATCH, SEED, DRY_*, THINK_ROW_SCALE,
     OUT.
"""
import os, re, json, torch

SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from e2e_qp_distill import build_student, BLOCK_SIZE
from loop_gate import DRYLogitsProcessor          # identical penalty to the gate
from datasets import load_dataset

ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
E2E = os.environ["E2E_MODEL"]
N_PROB = int(os.environ.get("N_PROB", "48"))
MAXNEW = int(os.environ.get("MAXNEW", "2048"))
THINK = os.environ.get("THINK", "1") == "1"
TEMP = float(os.environ.get("TEMP", "0.6"))
BATCH = int(os.environ.get("BATCH", "8"))
TRS = float(os.environ.get("THINK_ROW_SCALE", "1.0"))
THINK_CLOSE = 248069

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred(text):
    """Answer = last \\boxed{} if present, else the last number AFTER </think> (the answer block),
    else the last number anywhere. Matching the gate's convention that the committed answer is what
    follows the close tag."""
    b = text.rfind(r"\boxed")
    if b >= 0:
        j = text.find("{", b)
        if j >= 0:
            depth, k = 0, j
            for k in range(j, len(text)):
                depth += (text[k] == "{") - (text[k] == "}")
                if depth == 0:
                    break
            m = NUM.findall(text[j:k + 1])
            if m:
                return m[-1].replace(",", "").rstrip(".")
    tail = text.split("</think>")[-1] if "</think>" in text else text
    m = NUM.findall(tail) or NUM.findall(text)
    return m[-1].replace(",", "").rstrip(".") if m else None   # "42." -> "42"


def same(a, b):
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def main():
    tok = AutoTokenizer.from_pretrained(ORIG, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if os.environ.get("MODEL_KIND", "tern") == "fp":
        st = AutoModelForCausalLM.from_pretrained(E2E, trust_remote_code=True,
                                                  dtype=torch.bfloat16).to("cuda:0").eval()
    else:
        st, _ = build_student(E2E, ORIG, BLOCK_SIZE, "cuda:0")
        st.eval()
    st.config.use_cache = True

    # HEAD_MODE=fp|q4 swaps the ternary lm_head for a dense one (see head_swap.py). 'fp' is the
    # UPPER BOUND arm, not a deployable config: int4 is strictly worse than bf16, so a null result
    # there kills the head hypothesis without building the q4 arm.
    if os.environ.get("HEAD_MODE"):
        from head_swap import swap_head
        swap_head(st, os.environ.get("HEAD_SRC", "output_4bpipe/rotbase/modified_model"),
                  os.environ["HEAD_MODE"])
    if os.environ.get("EMBED_MODE"):        # the other half of the 24% (embed+head)
        from head_swap import swap_embed
        swap_embed(st, os.environ.get("HEAD_SRC", "output_4bpipe/rotbase/modified_model"),
                   os.environ["EMBED_MODE"])
    from head_swap import apply_think_gain      # identical on a ternary head (block scales) and a
    apply_think_gain(st, TRS, THINK_CLOSE)      # dense one (row weights) -- fold_think_scale.py

    if os.environ.get("DENSE_INFER", "0") == "1" and os.environ.get("MODEL_KIND", "tern") != "fp":
        from densify import densify, densify_selftest   # AFTER the TRS edit above -- densify
        densify_selftest(st)                            # snapshots scales, so order matters
        densify(st)

    gen = dict(max_new_tokens=MAXNEW, pad_token_id=tok.eos_token_id)
    gen.update(do_sample=True, temperature=TEMP, top_p=0.95, top_k=20) if TEMP > 0 \
        else gen.update(do_sample=False)
    dm = float(os.environ.get("DRY_MULT", "0"))
    if dm > 0:
        brk = []
        for t in ("\n", ":", '"', "*", ".", ","):
            ids = tok.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                brk.append(ids[0])
        gen["logits_processor"] = LogitsProcessorList([DRYLogitsProcessor(
            dm, float(os.environ.get("DRY_BASE", "1.75")),
            int(os.environ.get("DRY_ALLOWED", "2")), brk,
            int(os.environ.get("DRY_LAST_N", "0")))])
        print(f"DRY ON mult={dm}", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N_PROB))
    correct = closed = 0
    tlens, rows = [], []
    for i in range(0, N_PROB, BATCH):
        chunk = [ds[j] for j in range(i, min(i + BATCH, N_PROB))]
        msgs = [[{"role": "user", "content": r["question"]}] for r in chunk]
        try:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=THINK,
                                          return_tensors="pt", tokenize=True, padding=True,
                                          return_dict=True)
        except TypeError:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          tokenize=True, padding=True, return_dict=True)
        ii = enc["input_ids"].to("cuda:0")
        out = st.generate(ii, attention_mask=enc["attention_mask"].to("cuda:0"), **gen)
        for j, r in enumerate(chunk):
            g = out[j, ii.shape[1]:].tolist()
            txt = tok.decode(g, skip_special_tokens=False)
            gt, pr = gold(r["answer"]), pred(txt)
            ok = same(pr, gt)
            correct += ok
            if THINK_CLOSE in g:
                closed += 1
                tlens.append(g.index(THINK_CLOSE))
            rows.append({"gold": gt, "pred": pr, "ok": bool(ok), "n_tok": len(g),
                         "closed": THINK_CLOSE in g,
                         # Exact token ids, not just decoded text. Any sequence-level accumulation
                         # metric (ExAccErr, first-divergence position) must teacher-force the
                         # student's OWN token sequence, and decode -> re-encode is not guaranteed
                         # round-trip safe. Prompt ids are stored un-padded (left padding stripped
                         # via the attention mask) so the context can be rebuilt exactly.
                         "prompt_ids": enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist(),
                         "gen_ids": g, "text": txt})
        print(f"   {min(i+BATCH,N_PROB)}/{N_PROB}  acc={correct/min(i+BATCH,N_PROB):.3f}", flush=True)

    res = {"model": E2E, "n": N_PROB, "seed": SEED, "temp": TEMP, "think_row_scale": TRS,
           "dry_mult": dm, "accuracy": correct / N_PROB, "n_correct": correct,
           "closed_rate": closed / N_PROB,
           "mean_think_len": (sum(tlens) / len(tlens)) if tlens else 0.0}
    print("\n" + "=" * 60)
    print(f"  accuracy      {res['accuracy']:.4f}  ({correct}/{N_PROB})")
    print(f"  closed_rate   {res['closed_rate']:.4f}")
    print(f"  think_len     {res['mean_think_len']:.0f}")
    print("=" * 60)
    if os.environ.get("OUT"):
        json.dump(res, open(os.environ["OUT"], "w"), indent=1)
        if os.environ.get("ROWS"):
            json.dump(rows, open(os.environ["ROWS"], "w"), indent=1)


if __name__ == "__main__":
    main()
