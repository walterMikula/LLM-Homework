import argparse
import transformers
import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="google/flan-t5-small")
  parser.add_argument("--prompt", type=str, required=True)
  args = parser.parse_args()

  print("Loading model...")
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

  inputs = tokenizer(args.prompt, return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=50)

  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  print("Model Response:", response)

if __name__ == "__main__":
    main()
