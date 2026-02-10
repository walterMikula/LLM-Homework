import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

def normalize_answer(text):
    text = text.lower()
    if "true" in text:
        return "true"
    if "false" in text:
        return "false"
    return None

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

correct = 0
total = 0

with open("data/wiki_tf.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        statement = item["statement"]
        label = item["label"]

        prompt = f"Decide if this is true or false: {statement}. Answer only true or false."
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = normalize_answer(raw)

        print(f"Statement: {statement}")
        print(f"Expected: {label}, Model: {pred}")

        if pred == label:
            correct += 1
        total += 1

print(f"Accuracy: {correct}/{total}")
