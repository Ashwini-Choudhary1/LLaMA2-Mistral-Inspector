import json
import csv
import re
from pathlib import Path


RAW_OUTPUTS_PATH = Path("results/raw_outputs.jsonl")
BENCHMARK_RESULTS_PATH = Path("results/benchmark_results.csv")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_qa(output: str, reference: str) -> bool:
    output_norm = normalize(output)
    reference_norm = normalize(reference)

    ref_tokens = reference_norm.split()
    overlap = sum(token in output_norm for token in ref_tokens) / len(ref_tokens)

    return overlap >= 0.6


def extract_yes_no(text: str):
    text = text.lower()
    if text.strip().startswith("yes"):
        return "yes"
    if text.strip().startswith("no"):
        return "no"
    return None


def evaluate_reasoning(output: str, reference: str) -> bool:
    output_answer = extract_yes_no(output)
    reference_answer = extract_yes_no(reference)

    if output_answer is None or reference_answer is None:
        return False

    return output_answer == reference_answer


def evaluate_outputs():
    results = []

    with open(RAW_OUTPUTS_PATH, "r") as f:
        for line in f:
            record = json.loads(line)

            if record["task"] == "qa":
                correct = evaluate_qa(
                    output=record["output"],
                    reference=record["reference"]
                )

            elif record["task"] == "reasoning":
                correct = evaluate_reasoning(
                    output=record["output"],
                    reference=record["reference"]
                )

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
    from collections import defaultdict

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
