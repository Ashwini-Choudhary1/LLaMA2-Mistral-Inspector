import json
import time
from pathlib import Path
from models import load_model
from prompts import QA_PROMPT

DATA_PATH = Path("data/qa.jsonl")
OUTPUT_PATH = Path("results/raw_outputs.jsonl")

def load_dataset(path: Path):
    # for loading the dataset into list of dictionaries
    examples=[]

    with open(path,"r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def run_qa_benchmark():
    print("laoding model....")
    model = load_model("models/llama2.gguf")

    print("Loading QA Dataset")
    dataset = load_dataset(DATA_PATH)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print(f"Running QA benchmark on {len(dataset)}")

    with open(OUTPUT_PATH,"a") as out_f:
        for example in dataset:
            prompt= QA_PROMPT.format(input=example["input"])

            start = time.time()

            output = model(
                prompt,
                max_tokens=128,
                temperature=0.0
            )
            latency = time.time()- start

            response_text = output["choices"][0]["text"].strip()
            token_count= len(response_text.split())

            record = {
                "model": "Llama2",
                "task": "qa",
                "id": example["id"],
                "input": example["input"],
                "reference": example["reference"],
                "output": response_text,
                "latency_sec": round(latency, 2),
                "output_tokens": token_count
            }

            out_f.write(json.dumps(record) + "\n")

            print(f"[{example['id']}] {latency:.2f}s")

if __name__=="__main__":
    run_qa_benchmark()


