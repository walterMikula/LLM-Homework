from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

dataset = load_dataset("boolq", split="validation[:20]")

correct = 0
total = 0

for item in dataset:
    question = item["question"]
    answer = "yes" if item["answer"] else "no"

    prompt = f"Answer yes or no: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5)

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    pred = "yes" if "yes" in raw else "no"

    print(f"Question: {question}")
    print(f"Expected: {answer}, Model: {pred}")
    print("-----")

    if pred == answer:
        correct += 1
    total += 1

print(f"Accuracy: {correct}/{total}")
