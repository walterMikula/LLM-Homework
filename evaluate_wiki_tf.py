import argparse
import json
import re
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

PROMPT_TEMPLATE = """Decide whether the statement is true or false.
Answer with only "true" or "false".

Statement: {statement}
Answer:"""

def normalize_tf(raw: str):
    """Return 'true' or 'false' or None."""
    if raw is None:
        return None
    text = raw.strip().lower()

    # Most reliable: find the first true/false token
    m = re.search(r"\b(true|false)\b", text)
    if m:
        return m.group(1)

    # Common weird outputs from small models:
    # "t" / "f"
    if text.startswith("t"):
        return "true"
    if text.startswith("f"):
        return "false"

    # Sometimes they output 1/0
    if text == "1":
        return "true"
    if text == "0":
        return "false"

    return None

def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/flan-t5-small")
    parser.add_argument("--data", type=str, default="data/wiki_tf.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_incorrect", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    data = load_jsonl(args.data)

    correct = 0
    wrong = []

    for ex in data:
        prompt = PROMPT_TEMPLATE.format(statement=ex["statement"])
        inputs = tokenizer(prompt, return_tensors="pt")

        do_sample = args.temperature > 0.0
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
        )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred = normalize_tf(raw)
        gold = ex["label"].lower()

        if pred == gold:
            correct += 1
        else:
            wrong.append({
                "id": ex["id"],
                "statement": ex["statement"],
                "expected": gold,
                "raw": raw,
                "pred": pred
            })

    acc = correct / len(data) if data else 0.0
    print(f"Accuracy: {acc:.3f} ({correct}/{len(data)})")

    print("\nUp to 5 incorrect examples:")
    for item in wrong[: args.print_incorrect]:
        print(f'- id={item["id"]}')
        print(f'  statement: {item["statement"]}')
        print(f'  expected:  {item["expected"]}')
        print(f'  got(raw):  {item["raw"]}')
        print(f'  got(norm): {item["pred"]}')
        print()

if __name__ == "__main__":
    main()
