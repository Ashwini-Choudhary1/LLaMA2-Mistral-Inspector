from models import load_model
import time

def test_single_prompt():
    print("Loading model...")
    model = load_model("models/mistral.gguf")

    prompt = "Answer concisely: What is the capital of France?"

    print("Running inference...")
    start = time.time()

    output = model(
        prompt,
        max_tokens=50,
        stop=["\n"]
    )

    end = time.time()

    print("\nMODEL OUTPUT:")
    print(output["choices"][0]["text"].strip())
    print(f"\nLatency: {end - start:.2f} seconds")

if __name__ == "__main__":
    test_single_prompt()
