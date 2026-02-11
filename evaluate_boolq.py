import argparse
import re
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

PROMPT_A = """Read the passage and answer the question.
Answer only with "yes" or "no".
Passage:
{passage}
Question:
{question}
Answer:
"""

PROMPT_B = """Answer the question.
Answer only with "yes" or "no".
Question:
{question}
Answer:
"""


def normalize_yes_no(text: str):
    text = text.lower()
    m = re.search(r"\b(yes|no)\b", text)
    return m.group(1) if m else None


def ask_model(model, tokenizer, prompt: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def eval_experiment(data, model, tokenizer, with_passage: bool, max_new_tokens: int):
    correct = 0
    incorrect = []  # store up to 5 for req

    for ex in data:
        gold = "yes" if ex["answer"] else "no"
        if with_passage:
            prompt = PROMPT_A.format(passage=ex["passage"], question=ex["question"])
        else:
            prompt = PROMPT_B.format(question=ex["question"])

        raw = ask_model(model, tokenizer, prompt, max_new_tokens)
        pred = normalize_yes_no(raw)

        if pred == gold:
            correct += 1
        else:
            if len(incorrect) < 5:
                incorrect.append({
                    "question": ex["question"],
                    "expected": gold,
                    "raw": raw,
                    "pred": pred,
                    "passage": ex["passage"][:200] + ("..." if len(ex["passage"]) > 200 else "")
                })

    acc = correct / len(data) if len(data) > 0 else 0.0
    return acc, incorrect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/flan-t5-small")
    ap.add_argument("--n", type=int, default=100) 
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    ds = load_dataset("boolq", split="validation").shuffle(seed=args.seed)
    n = min(args.n, len(ds))
    data = ds.select(range(n))

    print(f"Subset size N = {n}")
    print()

    acc_a, wrong_a = eval_experiment(data, model, tok, with_passage=True, max_new_tokens=args.max_new_tokens)
    acc_b, wrong_b = eval_experiment(data, model, tok, with_passage=False, max_new_tokens=args.max_new_tokens)

    print(f"Experiment A (with passage) accuracy:    {acc_a:.4f}")
    print(f"Experiment B (without passage) accuracy: {acc_b:.4f}")
    print()

    print("Experiment A:")
    for ex in wrong_a:
        print("- question:", ex["question"])
        print("  expected:", ex["expected"])
        print("  pred:", ex["pred"])
        print("  raw:", ex["raw"])
        print("  passage_snippet:", ex["passage"])
        print()

    print("Experiment B:")
    for ex in wrong_b:
        print("- question:", ex["question"])
        print("  expected:", ex["expected"])
        print("  pred:", ex["pred"])
        print("  raw:", ex["raw"])
        print()


if __name__ == "__main__":
    main()
