# LLM Homework 1

## Downloading the libraries
Run:
pip install numpy transformers datasets sentencepiece torch


## Model Used
google/flan-t5-small

---

## How to Run

### Part 1 – Prompting Script- This script sends a prompt to the model and prints the response.
How to Run:
python llm_prompt.py --model google/flan-t5-small --prompt "Hello world" --max_new_tokens 128 --temperature 0.7 --seed 0
---

### Part 2 – Wikipedia True/False Evaluation- This script reads data/wiki_tf.jsonl and evaluates the model on 10 true/false statements about Charles Darwin.
How to run:
python evaluate_wiki_tf.py --model google/flan-t5-small --data ./data/wiki_tf.jsonl --temperature 0.0 --seed 0


---

### Part 3 – BoolQ Evaluation- This script evaluates the model on a subset of the BoolQ yes/no question dataset. 2 different experiments. Experiment A with passage and Experiment B without passage. 

Run:
python evaluate_boolq.py --model google/flan-t5-small --n 100 --max_new_tokens 8 --seed 0

---

## Accuracy Results for part 2 and 3

Wikipedia True/False Accuracy:
0.600 (6/10)

BoolQ Accuracy with n = 100
Experimetn A: 68% accuracy
Experiment B: 41% accuracy

---

## Discussion

The model performed okay on the small wiki true/false dataset but it also many many mistakes. On the BoolQ dataset, the accuracy was lower. I think thats becuse the questions were on a variety of topics and the model did not have direct context. It shows that small LLMs can get you answers but they might be incorrect and untrustworthy. Especially without information to support its descision making, the model will perform very bad. I would not use this on an exam, but for the sake of the assignment it at least runs the code lol.
