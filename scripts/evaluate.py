"""Standalone evaluation of a (LoRA) policy on the GSM8K test set.

Reports three numbers:
  * accuracy           — exact numeric match
  * partial_accuracy   — answer within 10% of ground truth
  * format_accuracy    — fraction of completions whose template parses

Run as:
    python evaluate.py
"""
import argparse
import json
import os

from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.sft.checkpoint_manager import CheckpointManager

from config import (
    GENERATION_CONFIGS,
    MAX_PROMPT_LENGTH,
    NUM_TEST_BATCHES,
    TEST_DATA_DIR,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    NUM_BATCHES,
    NUM_EPOCHS,
    DATA_SOURCE,
)
from data import SYSTEM_PROMPT, TEMPLATE, build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer, model_config_for
from rewards import match_format, match_numbers

DEFAULT_CKPT_ROOT = os.path.expanduser("~/tpu-2026/ckpts_backup/actor")


def restore_lora(lora_model, ckpt_root, step):
    """Restore LoRA adapter weights from an Orbax checkpoint (mirrors chat.py).

    As-shipped, evaluate.py never restored a checkpoint: it built a fresh LoRA
    model whose adapter is a no-op at init (B=0), so it only ever scored the
    *base* model. This helper lets us score a trained checkpoint.
    """
    mgr = CheckpointManager(root_directory=ckpt_root)
    n, _ = mgr.maybe_restore(model=lora_model, step=step, restore_only_lora_params=True)
    if n == 0:
        raise RuntimeError(
            f"No checkpoint found under {ckpt_root} (step={step}). "
            f"Check `ls {ckpt_root}`."
        )
    print(f"Restored LoRA params from step {n}")
    return n


def generate(question, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    if isinstance(question, str):
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)]
    else:
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q) for q in question]

    out = sampler(
        input_strings=batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature, top_k=top_k, top_p=top_p,
        echo=False, seed=seed, eos_tokens=eos_tokens,
    )
    return out.text[0] if isinstance(question, str) else out.text


def evaluate(dataset, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, num_passes=1):
    corr = partially_corr = corr_format = total = 0

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]
        per_q = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(questions, sampler, eos_tokens, temperature, top_k, top_p, seed=p)
            for i, r in enumerate(responses):
                per_q[i].append(r)

        for q, responses, ans in zip(questions, per_q, answers):
            got_corr = got_partial = got_format = False
            for r in responses:
                ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else "-1e9"
                try:
                    if float(ext.strip()) == float(ans.strip()):
                        got_corr = True
                    ratio = float(ext.strip()) / float(ans.strip())
                    if 0.9 <= ratio <= 1.1:
                        got_partial = True
                except Exception:
                    pass
                if match_format.search(r) is not None:
                    got_format = True
                if got_corr and got_partial and got_format:
                    break

            corr += int(got_corr)
            partially_corr += int(got_partial)
            corr_format += int(got_format)
            total += 1
            if total % 10 == 0:
                print(f"===> corr={corr} total={total} acc={corr/total*100:.2f}% "
                      f"partial={partially_corr/total*100:.2f}% fmt={corr_format/total*100:.2f}%")

    return corr, total, corr/total*100, partially_corr/total*100, corr_format/total*100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="greedy", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_ROOT,
                    help=f"Directory of per-step LoRA checkpoint subdirs. Default: {DEFAULT_CKPT_ROOT}")
    ap.add_argument("--steps", default="",
                    help="Comma-separated checkpoint steps to evaluate, e.g. '3364' or "
                         "'2000,3000,3364'. '0' = latest. Empty = base model only.")
    ap.add_argument("--include-base", action="store_true",
                    help="Also evaluate the base model (no adapter) alongside the checkpoint(s).")
    ap.add_argument("--no-restore", action="store_true",
                    help="Evaluate the base model only (ignore --steps). As-shipped behaviour.")
    ap.add_argument("--num-batches", type=int, default=NUM_TEST_BATCHES,
                    help=f"Number of GSM8K-test micro-batches (prompts). Default: {NUM_TEST_BATCHES}.")
    ap.add_argument("--out", default=None, help="Optional path to dump a results JSON.")
    args = ap.parse_args()

    steps = [] if args.no_restore else [int(s) for s in args.steps.split(",") if s.strip()]
    # The base model must be scored BEFORE any restore: maybe_restore() mutates the
    # LoRA weights in place, and at init the adapter is a no-op so the un-restored
    # LoRA graph is exactly the base model. Score base first whenever requested.
    run_base = args.no_restore or args.include_base or not steps

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES, args.num_batches, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
    )

    sampler = sampler_lib.Sampler(
        transformer=lora,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )

    preset = GENERATION_CONFIGS[args.preset]
    results = []

    if run_base:
        print("\n==== Evaluating BASE model (gemma-3-1b-it, no adapter) ====")
        n, t, acc, pacc, facc = evaluate(test_ds, sampler, eos_tokens, **preset)
        print(f"BASE  correct={n}/{t}  acc={acc:.2f}%  partial={pacc:.2f}%  format={facc:.2f}%")
        results.append({"model": "base", "step": None, "n": n, "total": t,
                        "acc": acc, "partial": pacc, "format": facc})

    for st in steps:
        restore_lora(lora, args.ckpt_dir, None if st == 0 else st)
        print(f"\n==== Evaluating LoRA checkpoint step {st} ====")
        n, t, acc, pacc, facc = evaluate(test_ds, sampler, eos_tokens, **preset)
        print(f"STEP {st}  correct={n}/{t}  acc={acc:.2f}%  partial={pacc:.2f}%  format={facc:.2f}%")
        results.append({"model": "lora", "step": st, "n": n, "total": t,
                        "acc": acc, "partial": pacc, "format": facc})

    print(f"\n========== SUMMARY  (GSM8K {args.source} test, "
          f"{args.num_batches} prompts, preset={args.preset}) ==========")
    print(f"{'model':16s} {'exact%':>8s} {'within10%':>10s} {'format%':>8s}")
    for r in results:
        name = "base" if r["step"] is None else f"step {r['step']}"
        print(f"{name:16s} {r['acc']:8.2f} {r['partial']:10.2f} {r['format']:8.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"preset": args.preset, "source": args.source,
                       "num_prompts": args.num_batches, "ckpt_dir": args.ckpt_dir,
                       "results": results}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
