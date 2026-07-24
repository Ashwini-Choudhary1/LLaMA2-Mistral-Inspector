import json
import time
import argparse
from pathlib import Path
from models import load_model
import prompts
from config import MODELS, TASKS, GENERATION_CONFIG

OUTPUT_PATH = Path("results/raw_outputs.jsonl")

def load_dataset(path: Path):
    examples=[]
    if not path.exists():
        print(f"Dataset path {path} does not exist.")
        return examples
    with open(path,"r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def run_benchmark(model_name: str, task_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in config.")
    if task_name not in TASKS:
        raise ValueError(f"Task {task_name} not found in config.")

    model_config = MODELS[model_name]
    task_config = TASKS[task_name]

    print(f"Loading model {model_name} from {model_config['path']}....")
    try:
        model = load_model(str(model_config["path"]))
    except FileNotFoundError:
        print(f"Model file not found. Ensure it exists at {model_config['path']}")
        return

    print(f"Loading Dataset for task {task_name}")
    dataset = load_dataset(task_config["data_path"])

    if not dataset:
        return

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print(f"Running benchmark on {len(dataset)} examples")
    
    prompt_template_name = task_config["prompt_template"]
    prompt_template = getattr(prompts, prompt_template_name)

    temp = GENERATION_CONFIG["temperature"]
    top_p = GENERATION_CONFIG["top_p"]
    max_tokens = GENERATION_CONFIG["max_new_tokens"]

    with open(OUTPUT_PATH,"a") as out_f:
        for example in dataset:
            prompt = prompt_template.format(input=example["input"])

            start = time.time()
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop=GENERATION_CONFIG.get("stop", [])
            )
            latency = time.time() - start

            response_text = output["choices"][0]["text"].strip()
            token_count = len(response_text.split())

            record = {
                "model": model_name,
                "task": task_name,
                "id": example["id"],
                "input": example["input"],
                "reference": example["reference"],
                "output": response_text,
                "latency_sec": round(latency, 2),
                "output_tokens": token_count,
                "temperature": temp,
                "top_p": top_p,
                "max_new_tokens": max_tokens
            }

            out_f.write(json.dumps(record) + "\n")
            print(f"[{example['id']}] {latency:.2f}s")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run LLM Benchmarks")
    parser.add_argument("--model", type=str, default="llama2", help="Model name (e.g., llama2, mistral)")
    parser.add_argument("--task", type=str, default="reasoning", help="Task name (e.g., qa, reasoning, summarization)")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.task)
