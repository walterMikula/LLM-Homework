import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/flan-t5-small")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Set random seeds for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt")

    do_sample = args.temperature > 0.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model Response:", response)

if __name__ == "__main__":
    main()
