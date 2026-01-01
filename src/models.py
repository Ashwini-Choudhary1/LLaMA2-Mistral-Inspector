from pathlib import Path
from llama_cpp import Llama

def load_model(model_path: str):
    model_path= Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found at path {model_path}")
    
    llm = Llama(

        model_path= str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False

    )

    return llm

