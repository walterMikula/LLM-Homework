# LLM Homework 1

## Model Used
google/flan-t5-small

---

## How to Run

### Part 1 – Prompting Script- This script sends a prompt to the model and prints the response.
Run:
python llm_prompt.py --prompt "Hello" --max_new_tokens 10 --temperature 0.0 --seed 0

---

### Part 2 – Wikipedia True/False Evaluation- This script reads data/wiki_tf.jsonl and evaluates the model on 10 true/false statements about Charles Darwin.
Run:
python evaluate_wiki_tf.py

---

### Part 3 – BoolQ Evaluation- This script evaluates the model on a subset of the BoolQ yes/no question dataset.

Run:
python evaluate_boolq.py

---

## Results

Wikipedia True/False Accuracy:
0.600 (6/10)

BoolQ Accuracy:
5/20

---
