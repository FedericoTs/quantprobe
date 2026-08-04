"""S-1 phases 2-3: LoRA-tune Qwen3-0.6B on the filtered teacher pairs, score against the stakes.

  python weights/s1_train.py            # eval base -> train -> eval tuned -> verdicts

Design decisions, all staked or K3-informed (see preregistrations/2026-08-04-s1 + NOTES_K3):
- LOW-EFFORT student: chat template applied with enable_thinking=False for both training
  targets and evaluation. The 0.6B cannot host long reasoning; we train the direct-answer
  specialist deliberately.
- Manual training loop (no trl): fewer moving parts, seeded, deterministic-enough for a
  first arm. bf16 unsupported on Pascal -> fp16 autocast with GradScaler.
- The SAME s1_gen.check that filtered the teacher scores the student. One checker, all arms.
"""
from __future__ import annotations
import json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s1_gen import SPLITS, check, instance  # noqa: E402

MODEL_DIR = "D:/evo-compress-data/hf/Qwen3-0.6B"
OUT_DIR = "D:/evo-compress-data/hf/s1-adapter"
DATA = os.path.join(HERE, "data")
SEED = 20260804


def load_pairs():
    rows = json.load(open(os.path.join(DATA, "s1_train_teacher.json"), encoding="utf-8"))
    pairs = [(r["prompt"], r["teacher"]) for r in rows if r["passed"]]
    print(f"training pairs (teacher-clean only): {len(pairs)}")
    return pairs


def build_model(train=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16,
                                                 device_map="cuda")
    if train:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return tok, model


def encode(tok, prompt, answer=None):
    """Chat-templated ids; labels mask the prompt. enable_thinking=False everywhere."""
    msgs = [{"role": "user", "content": prompt}]
    # tokenize=True returns a tokenizers.Encoding on this transformers version, which
    # torch.tensor cannot ingest - go via the template STRING and tokenize explicitly,
    # which yields a plain list[int] on every version.
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
    if answer is None:
        return prompt_ids
    full = prompt_ids + tok(answer, add_special_tokens=False)["input_ids"] \
           + [tok.eos_token_id]
    labels = [-100] * len(prompt_ids) + full[len(prompt_ids):]
    return full, labels


def evaluate(tok, model, tag):
    import torch
    model.eval()
    out = {}
    for split, seeds in (("heldout", range(*SPLITS["heldout"])),
                         ("control", range(*SPLITS["control"]))):
        good = total = 0
        t0 = time.time()
        for sd in seeds:
            ins = instance(sd)
            ids = encode(tok, ins["prompt"])
            x = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=256, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            txt = tok.decode(y[0][len(ids):], skip_special_tokens=True)
            good += bool(check(ins["truth"], txt))
            total += 1
        out[split] = {"clean": good, "total": total, "rate": good / total,
                      "seconds": round(time.time() - t0, 1)}
        print(f"  [{tag}] {split}: {good}/{total} = {100*good/total:.1f}% "
              f"({out[split]['seconds']}s)", flush=True)
    return out


def train(tok, model, pairs):
    import torch
    random.Random(SEED).shuffle(pairs)
    torch.manual_seed(SEED)
    enc = [encode(tok, p, a) for p, a in pairs]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3 * len(enc))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    accum, step = 8, 0
    for epoch in range(3):
        tot = n = 0.0
        for i, (ids, labels) in enumerate(enc):
            x = torch.tensor([ids], device="cuda")
            y = torch.tensor([labels], device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y).loss / accum
            scaler.scale(loss).backward()
            tot += loss.item() * accum
            n += 1
            if (i + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
        print(f"  epoch {epoch+1}/3  mean loss {tot/n:.4f}  ({step} opt steps)", flush=True)
    model.save_pretrained(OUT_DIR)
    print(f"adapter saved -> {OUT_DIR}")


def main():
    pairs = load_pairs()
    print("\n=== eval: BASE student ===", flush=True)
    tok, base = build_model(train=False)
    base_scores = evaluate(tok, base, "base")
    del base
    import torch
    torch.cuda.empty_cache()

    print("\n=== train: LoRA on filtered teacher pairs ===", flush=True)
    tok, model = build_model(train=True)
    train(tok, model, pairs)

    print("\n=== eval: TUNED student ===", flush=True)
    tuned_scores = evaluate(tok, model, "tuned")

    teacher_ho = 116 / 120                       # measured phase 1, the staked P2 bar
    b_ho, t_ho = base_scores["heldout"]["rate"], tuned_scores["heldout"]["rate"]
    b_c, t_c = base_scores["control"]["rate"], tuned_scores["control"]["rate"]
    p1 = (t_ho - b_ho) >= 0.10
    p2 = t_ho >= teacher_ho
    kra_fired = (b_c - t_c) > 0.15
    verdict = {
        "base": base_scores, "tuned": tuned_scores, "teacher_heldout_bar": teacher_ho,
        "P1_gain_ge_10pts": {"gain": round(t_ho - b_ho, 4), "pass": p1},
        "P2_ge_teacher": {"tuned": round(t_ho, 4), "bar": round(teacher_ho, 4), "pass": p2},
        "KRA_forgetting_gt_15pts": {"drop": round(b_c - t_c, 4), "fired": kra_fired},
    }
    with open(os.path.join(DATA, "s1_student_results.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    print("\n=== STAKED VERDICTS ===")
    print(f"  P1  (tuned - base >= +10pts on held-out): {t_ho-b_ho:+.1%}  ->",
          "PASS" if p1 else "FAIL")
    print(f"  P2  (tuned >= teacher {teacher_ho:.1%}):   {t_ho:.1%}  ->",
          "PASS" if p2 else "FAIL")
    print(f"  KR-A (control drop > 15pts kills):        {b_c-t_c:+.1%} drop ->",
          "FIRED - S-1 FAILS" if kra_fired else "holds")
    return 1 if kra_fired else 0


if __name__ == "__main__":
    sys.exit(main())
