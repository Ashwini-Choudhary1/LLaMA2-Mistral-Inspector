import json
import time
from pathlib import Path
from models import load_model
from prompts import REASONING_PROMPT #changeable parameter

DATA_PATH = Path("data/reasoning.jsonl") #changeable parameter
OUTPUT_PATH = Path("results/raw_outputs.jsonl")

def load_dataset(path: Path):
    # for loading the dataset into list of dictionaries
    examples=[]

    with open(path,"r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def run_qa_benchmark():
    print("loading model....")
    model = load_model("models/llama2.gguf") #changeable parameter

    print("Loading Dataset")
    dataset = load_dataset(DATA_PATH)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print(f"Running benchmark on {len(dataset)}")
    TEMPERATURE = 0.5 #changeable parameter
    TOP_P = 0.9 #changeable parameter
    MAX_NEW_TOKENS = 128 #changeable parameter


    with open(OUTPUT_PATH,"a") as out_f:
        for example in dataset:
            prompt= REASONING_PROMPT.format(input=example["input"])

            start = time.time()

            output = model(
                prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            latency = time.time()- start

            response_text = output["choices"][0]["text"].strip()
            token_count= len(response_text.split())

            record = {
            "model": "Llama2", #changeable parameter
            "task": "reasoning",       #changeable parameter
            "id": example["id"],
            "input": example["input"],
            "reference": example["reference"],
            "output": response_text,
            "latency_sec": round(latency, 2),
            "output_tokens": token_count,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS
            }


            out_f.write(json.dumps(record) + "\n")

            print(f"[{example['id']}] {latency:.2f}s")

if __name__=="__main__":
    run_qa_benchmark()


