from pathlib import Path

GENERATION_CONFIG = {
    "temperature": 0.5,
    "max_new_tokens": 128,
    "top_p": 0.9,
    "stop": ["\n\n", "Question:", "Problem:", "Text:"]
}

MODELS = {
    "llama2": {
        "name": "llama2",
        "type": "gguf",
        "path": Path("models/llama2.gguf")
    },
    "mistral": {
        "name": "mistral",
        "type": "gguf",
        "path": Path("models/mistral.gguf")
    }
}

TASKS = {
    "qa": {
        "data_path": Path("data/qa.jsonl"),
        "prompt_template": "QA_PROMPT"
    },
    "reasoning": {
        "data_path": Path("data/reasoning.jsonl"),
        "prompt_template": "REASONING_PROMPT"
    },
    "summarization": {
        "data_path": Path("data/summarization.jsonl"),
        "prompt_template": "SUMMARIZATION_PROMPT"
    }
}
