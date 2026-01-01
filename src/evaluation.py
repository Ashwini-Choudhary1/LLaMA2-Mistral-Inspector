import json
import csv
import re
from pathlib import Path


RAW_OUTPUTS_PATH = Path("results/raw_outputs.jsonl")
BENCHMARK_RESULTS_PATH = Path("results/benchmark_results.csv")


def normalize(text: str) -> str:
    
    # Normalize text for comparison:
    #lowercase
    #removed punctuation
    #normalized whitespace
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(output: str, reference: str) -> bool:
    
    norm_output = normalize(output)
    norm_reference = normalize(reference)

    ref_tokens = norm_reference.split()
    matched_tokens = sum(token in norm_output for token in ref_tokens)

    return (matched_tokens / len(ref_tokens)) >= 0.6


def evaluate_qa_outputs():
    results = []

    with open(RAW_OUTPUTS_PATH, "r") as f:
        for line in f:
            record = json.loads(line)

            record["correct"] = is_correct(
                output=record["output"],
                reference=record["reference"]
            )

            results.append(record)

    return results


def write_summary_csv(results):
    BENCHMARK_RESULTS_PATH.parent.mkdir(exist_ok=True)

    with open(BENCHMARK_RESULTS_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "model",
                "task",
                "id",
                "latency_sec",
                "output_tokens",
                "correct"
            ]
        )

        writer.writeheader()

        for r in results:
            writer.writerow({
                "model": r["model"],
                "task": r["task"],
                "id": r["id"],
                "latency_sec": r["latency_sec"],
                "output_tokens": r["output_tokens"],
                "correct": r["correct"]
            })


def print_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(r["latency_sec"] for r in results) / total
    avg_tokens = sum(r["output_tokens"] for r in results) / total

    print("\n--- QA Evaluation Summary ---")
    print(f"Total examples : {total}")
    print(f"Accuracy       : {accuracy:.2%}")
    print(f"Avg latency    : {avg_latency:.2f}s")
    print(f"Avg tokens     : {avg_tokens:.2f}")


if __name__ == "__main__":
    results = evaluate_qa_outputs()
    write_summary_csv(results)
    print_metrics(results)
