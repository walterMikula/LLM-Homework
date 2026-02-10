# LLM Homework 1

## Model Used
google/flan-t5-small

---

## How to Run

### Part 1 – Prompting Script
Run:
python llm_prompt.py --prompt "Hello" --max_new_tokens 10 --temperature 0.0 --seed 0

This script sends a prompt to the model and prints the response.

---

### Part 2 – Wikipedia True/False Evaluation
Run:
python evaluate_wiki_tf.py

This script reads data/wiki_tf.jsonl and evaluates the model on 10 true/false statements about Charles Darwin.

---

### Part 3 – BoolQ Evaluation
Run:
python evaluate_boolq.py

This script evaluates the model on a subset of the BoolQ yes/no question dataset.

---

## Results

Wikipedia True/False Accuracy:
0.600 (6/10)

BoolQ Accuracy:
5/20

---
