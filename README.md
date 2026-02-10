First i ran the following command to install the libraries:
$pip install transformers datasets torch sentencepiece

The model that I used is google/flan-t5-small
###
How to Run:

Run:
python llm_prompt.py --prompt "Hello" --max_new_tokens 10 --temperature 0.0 --seed 0

This script will send a prompt to the model and print the response.

Part 2: Evaluating wikipedia true and false questions:
Run: python evaluate_wiki_tf.py

This script reads data/wiki_tf.jsonl and evaluates the model on 10 true/false statements about Charles Darwin.
