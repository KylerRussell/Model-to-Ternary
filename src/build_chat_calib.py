#!/usr/bin/env python
"""build_chat_calib.py — Stage-1 chat/reasoning-format calibration corpus (researcher round-2, item 1).

The generic calib has ~ZERO `assistant\n<think>\n` boundaries, so block-AP never reconstructs, and E2E never
supervises, the start-of-thinking context where the ternary model collapses (M3 argmax-agree = 0%). This builds
a corpus SATURATED with that boundary: FP thinking-mode rollouts over a diverse prompt set, packed so each
1024-tok training sequence carries several `<think>\n` boundaries. We MIX in a replay fraction of the generic
calib so block-AP/E2E keep general ability (avoid catastrophic forgetting).

Output = JSON list of token-id lists (>= --seq), consumed identically by block_ap_recovery (calibration_data)
and e2e_qp_distill (--calib) + its teacher-cache builder.

  python build_chat_calib.py --model output_4b/untied_4b --out output_4b/chat_calib.json \
     --n-rollouts 512 --max-new 512 --seq 1024 --replay output_4b/test1_data/calib_16M.json --replay-frac 0.5
"""
import argparse, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gen_reasoning_traces import PROMPTS as REASON_PROMPTS

THINK_OPEN, THINK_CLOSE, EOS = 248068, 248069, 248046

PLAIN = [
    "What is the capital of France?", "Write a haiku about the ocean.",
    "Explain photosynthesis simply.", "Who wrote Pride and Prejudice?",
    "Give me three tips for better sleep.", "Summarize the water cycle.",
    "What causes the seasons?", "Recommend a book for a rainy afternoon.",
    "Describe how a bill becomes law.", "What is compound interest?",
    "How do vaccines work?", "Explain the difference between weather and climate.",
    "What is the Pythagorean theorem used for?", "How does a refrigerator keep food cold?",
    "Name three renewable energy sources and how they work.", "What is the role of mitochondria?",
]

# EASY = short-answer prompts that reliably CLOSE </think> fast -> dense post_think/answer boundaries.
EASY = [
    "What is 2+2?", "What is 17 times 3?", "What is 100 divided by 4?", "What is 9 squared?",
    "What is the capital of Japan?", "What is the capital of Italy?", "What is the capital of Canada?",
    "What color do you get mixing blue and yellow?", "How many days are in a week?",
    "How many continents are there?", "What is the chemical symbol for water?",
    "What is the largest planet in our solar system?", "What is the freezing point of water in Celsius?",
    "Who painted the Mona Lisa?", "What is the square root of 64?", "What is 15% of 200?",
    "How many sides does a hexagon have?", "What is the opposite of 'hot'?",
    "What gas do plants absorb from the air?", "What is the tallest mountain on Earth?",
    "How many legs does a spider have?", "What is the boiling point of water in Celsius?",
    "What is 7 plus 8?", "What is the plural of 'mouse'?", "What is the speed of light approximately?",
    "Name the primary colors.", "What is the smallest prime number?", "What year did World War II end?",
    "What is the currency of the United States?", "How many minutes are in an hour?",
    "What is 12 times 12?", "What is the chemical symbol for gold?", "What is half of 50?",
    "What planet is known as the Red Planet?", "What is the largest ocean on Earth?",
    "How many letters are in the English alphabet?", "What is the sum of angles in a triangle?",
    "What is 3 to the power of 4?", "Who wrote Romeo and Juliet?", "What is the capital of Germany?",
]


def build_prompts(tok, replay_path, n_derived):
    """returns (hard_pool, easy_pool). hard = diverse/long-CoT; easy = short-answer (close </think> fast)."""
    hard = list(REASON_PROMPTS) + PLAIN
    if replay_path and n_derived > 0:
        gen = json.load(open(replay_path))
        random.shuffle(gen)
        for s in gen[:n_derived]:
            txt = tok.decode(s[:40]).strip().replace("\n", " ")
            if len(txt) > 20:
                hard.append(f"Continue and explain: {txt}")
    random.shuffle(hard)
    return hard, list(EASY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output_4b/untied_4b")
    ap.add_argument("--out", default="output_4b/chat_calib.json")
    ap.add_argument("--n-rollouts", type=int, default=512)
    ap.add_argument("--n-derived", type=int, default=200, help="extra prompts derived from replay text")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--replay", default="output_4b/test1_data/calib_16M.json")
    ap.add_argument("--replay-frac", type=float, default=0.5, help="fraction of FINAL packed seqs from replay")
    ap.add_argument("--require-close", action="store_true",
                    help="keep only rollouts that CLOSED </think> (dense post_think/answer boundaries)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (shard with different seeds)")
    ap.add_argument("--device", default="cuda:0", help="GPU for this shard")
    # A plain .to(device) load caps this script at models that fit on ONE card (fine at 4B = 11.8 GiB,
    # impossible at 27B = ~54 GiB bf16). --device-map hands the placement to accelerate so the FP teacher
    # can be split across both GPUs + CPU. Sharding (one whole model per GPU) is faster when it fits, so
    # the caller should prefer --device and fall back to --device-map only when it does not.
    ap.add_argument("--device-map", default=None,
                    help="accelerate device_map (e.g. 'auto') for models too large for one GPU; "
                         "overrides --device")
    ap.add_argument("--gpu-mem", default="20GiB", help="per-GPU cap when --device-map is used")
    ap.add_argument("--cpu-mem", default="30GiB", help="CPU offload cap when --device-map is used")
    ap.add_argument("--easy-frac", type=float, default=0.0,
                    help="fraction of rollouts drawn from EASY short-answer prompts (raise </think> close-rate)")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"loading FP teacher {args.model} ...", flush=True)
    if args.device_map:
        mm = {i: args.gpu_mem for i in range(torch.cuda.device_count())}
        mm["cpu"] = args.cpu_mem
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                     dtype=torch.bfloat16,
                                                     device_map=args.device_map, max_memory=mm).eval()
        dev = model.device          # accelerate puts the input embedding here; hooks move the rest
        print(f"  device_map={args.device_map} max_memory={mm} -> inputs on {dev}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                     dtype=torch.bfloat16).to(args.device).eval()
        dev = args.device

    hard, easy = build_prompts(tok, args.replay, args.n_derived)
    n_easy = int(args.n_rollouts * args.easy_frac)
    plist = [easy[i % len(easy)] for i in range(n_easy)] + \
            [hard[i % len(hard)] for i in range(args.n_rollouts - n_easy)]
    random.shuffle(plist)
    print(f"{len(hard)} hard + {len(easy)} easy prompts -> {args.n_rollouts} rollouts "
          f"({n_easy} easy, batch {args.batch})", flush=True)

    rollouts, n_valid = [], 0
    for b0 in range(0, len(plist), args.batch):
        chunk = plist[b0:b0 + args.batch]
        msgs = [[{"role": "user", "content": p}] for p in chunk]
        try:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                                          return_tensors="pt", tokenize=True, padding=True, return_dict=True)
        except TypeError:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          tokenize=True, padding=True, return_dict=True)
        input_ids = enc["input_ids"].to(dev)
        attn = enc["attention_mask"].to(dev)
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attn, max_new_tokens=args.max_new,
                                 do_sample=True, temperature=args.temp, top_p=args.top_p, top_k=20,
                                 pad_token_id=tok.eos_token_id)
        for j in range(out.shape[0]):
            row = out[j].tolist()
            # strip left padding: real sequence starts at first non-pad in the prompt region
            am = attn[j].tolist()
            start = am.index(1) if 1 in am else 0
            full = row[start:]
            # trim trailing pad/eos repeats but keep one eos
            while len(full) > 2 and full[-1] == tok.eos_token_id and full[-2] == tok.eos_token_id:
                full.pop()
            rollouts.append(full)
            n_valid += (THINK_CLOSE in full)
        if (b0 // args.batch + 1) % 4 == 0:
            print(f"  {b0+len(chunk)}/{len(plist)} rollouts | closed-think {n_valid}", flush=True)

    avg = sum(len(r) for r in rollouts) / len(rollouts)
    print(f"generated {len(rollouts)} rollouts, avg len {avg:.0f}, closed-think {n_valid}", flush=True)

    if args.require_close:
        rollouts = [r for r in rollouts if THINK_CLOSE in r]
        print(f"require-close: kept {len(rollouts)} closed-think rollouts", flush=True)

    # ---- pack rollouts into >= seq sequences (dense <think> boundaries) ----
    random.shuffle(rollouts)
    chat_seqs, buf = [], []
    for r in rollouts:
        buf.extend(r)
        while len(buf) >= args.seq:
            chat_seqs.append(buf[:args.seq])
            buf = buf[args.seq:]
    print(f"packed -> {len(chat_seqs)} chat seqs of {args.seq}", flush=True)

    # ---- mix replay ----
    final = list(chat_seqs)
    if args.replay and args.replay_frac > 0 and len(chat_seqs) > 0:
        n_replay = int(len(chat_seqs) * args.replay_frac / (1 - args.replay_frac))
        gen = [s[:args.seq] for s in json.load(open(args.replay)) if len(s) >= args.seq]
        random.shuffle(gen)
        final += gen[:n_replay]
        print(f"mixed {min(n_replay,len(gen))} replay seqs (target frac {args.replay_frac})", flush=True)
    random.shuffle(final)

    json.dump(final, open(args.out, "w"))
    boundaries = sum(s.count(THINK_OPEN) for s in final)
    print(f"\nwrote {len(final)} seqs ({len(final)*args.seq/1e6:.2f}M tokens), "
          f"{boundaries} <think> boundaries -> {args.out}")


if __name__ == "__main__":
    main()
