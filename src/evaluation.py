import json
import csv
import re
from pathlib import Path
from collections import defaultdict


RAW_OUTPUTS_PATH = Path("results/raw_outputs.jsonl")
BENCHMARK_RESULTS_PATH = Path("results/benchmark_results.csv")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def evaluate_qa(output: str, reference: str) -> bool:
    output_n = normalize(output)
    reference_n = normalize(reference)
    return token_overlap(reference_n, output_n) >= 0.6


def extract_yes_no(text: str):
    text = text.lower().strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def evaluate_reasoning(output: str, reference: str) -> bool:
    return extract_yes_no(output) == extract_yes_no(reference)


def evaluate_summarization(output: str, reference: str) -> bool:
    output_n = normalize(output)
    reference_n = normalize(reference)
    return token_overlap(reference_n, output_n) >= 0.5


def evaluate_outputs():
    results = []

    with open(RAW_OUTPUTS_PATH, "r") as f:
        for line in f:
            record = json.loads(line)

            if record["task"] == "qa":
                correct = evaluate_qa(record["output"], record["reference"])

            elif record["task"] == "reasoning":
                correct = evaluate_reasoning(record["output"], record["reference"])

            elif record["task"] == "summarization":
                correct = evaluate_summarization(record["output"], record["reference"])

            else:
                continue

            record["correct"] = correct
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
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["task"])].append(r)

    print("\n--- Evaluation Summary ---")
    for (model, task), rows in grouped.items():
        accuracy = sum(r["correct"] for r in rows) / len(rows)
        print(f"{model} | {task} | Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    results = evaluate_outputs()
    write_summary_csv(results)
    print_metrics(results)
