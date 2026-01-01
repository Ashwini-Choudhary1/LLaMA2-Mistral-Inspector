import json
import csv
from pathlib import Path


RAW_OUTPUTS_PATH = Path("results/raw_outputs.jsonl")
FAILURE_SUMMARY_PATH = Path("results/failure_summary.csv")


def extract_yes_no(text: str):
    text = text.lower().strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def token_overlap(reference: str, output: str) -> float:
    ref_tokens = set(reference.lower().split())
    out_tokens = set(output.lower().split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & out_tokens) / len(ref_tokens)


def classify_qa_failure(output: str, reference: str) -> str:
    overlap = token_overlap(reference, output)
    output_tokens = len(output.split())
    reference_tokens = len(reference.split())

    if overlap >= 0.6:
        if output_tokens > reference_tokens * 2:
            return "over_verbose_correct"
        return "correct"

    if any(word in output.lower() for word in ["unknown", "not sure", "cannot determine"]):
        return "partial_answer"

    if output_tokens > reference_tokens * 2:
        return "hallucination"

    if output.strip().isalpha() and output.lower() not in reference.lower():
        return "wrong_entity"

    return "overconfident_wrong"


def classify_reasoning_failure(output: str, reference: str) -> str:
    output_answer = extract_yes_no(output)
    reference_answer = extract_yes_no(reference)

    if output_answer is None or reference_answer is None:
        return "unverifiable_reasoning"

    if output_answer != reference_answer:
        return "invalid_logical_inference"

    return "correct"


def classify_summarization_failure(output: str, reference: str) -> str:
    overlap = token_overlap(reference, output)
    output_tokens = len(output.split())
    reference_tokens = len(reference.split())

    if overlap >= 0.5:
        return "correct"

    if output_tokens > reference_tokens * 2:
        return "hallucinated_detail"

    return "missing_key_info"


def analyze_failures():
    failures = []

    with open(RAW_OUTPUTS_PATH, "r") as f:
        for line in f:
            record = json.loads(line)

            if record["task"] == "qa":
                failure_type = classify_qa_failure(record["output"], record["reference"])
                if failure_type not in ["correct", "over_verbose_correct"]:
                    failures.append({**record, "failure_type": failure_type})

            elif record["task"] == "reasoning":
                failure_type = classify_reasoning_failure(record["output"], record["reference"])
                if failure_type != "correct":
                    failures.append({**record, "failure_type": failure_type})

            elif record["task"] == "summarization":
                failure_type = classify_summarization_failure(record["output"], record["reference"])
                if failure_type != "correct":
                    failures.append({**record, "failure_type": failure_type})

    return failures


def write_failure_summary(failures):
    FAILURE_SUMMARY_PATH.parent.mkdir(exist_ok=True)

    with open(FAILURE_SUMMARY_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "model",
                "task",
                "id",
                "failure_type",
                "output",
                "reference"
            ]
        )

        writer.writeheader()
        for f in failures:
            writer.writerow({
                "model": f["model"],
                "task": f["task"],
                "id": f["id"],
                "failure_type": f["failure_type"],
                "output": f["output"],
                "reference": f["reference"]
            })


if __name__ == "__main__":
    failures = analyze_failures()
    write_failure_summary(failures)
    print("\n--- Failure Analysis Summary ---")
    print(f"Total failures found: {len(failures)}")
