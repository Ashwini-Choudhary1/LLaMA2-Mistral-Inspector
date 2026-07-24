import sys
import time
from pathlib import Path

# Add src to path so we can import from there
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from config import MODELS, TASKS, GENERATION_CONFIG
import prompts
from models import load_model
from evaluation import evaluate_qa, evaluate_reasoning, evaluate_summarization
from failure_analysis import classify_qa_failure, classify_reasoning_failure, classify_summarization_failure

app = FastAPI(title="LLaMA2-Mistral-Inspector API")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class GenerateRequest(BaseModel):
    model_name: str
    task_name: str
    input_text: str
    reference_text: str

# Cache for loaded models
loaded_models = {}

def get_model(model_name: str):
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    if model_name in loaded_models:
        return loaded_models[model_name]
        
    model_path = MODELS[model_name]["path"]
    
    try:
        model = load_model(str(model_path))
        loaded_models[model_name] = model
        return model
    except FileNotFoundError:
        print(f"Warning: Model {model_name} not found at {model_path}. Using MOCK model.")
        
        # Define a mock model for UI testing
        class MockModel:
            def __call__(self, prompt, max_tokens, temperature, top_p):
                # Simulate generation time
                time.sleep(1.5)
                # Create a fake response based on task type to show failures/successes
                if "Summarize" in prompt:
                    text = "This is a mocked summary that might be missing key info, because the models weren't downloaded."
                elif "Problem:" in prompt:
                    text = "Explanation: Step 1. I am a mock model. Therefore, Yes."
                else:
                    text = "This is a highly verbose mock answer that probably triggers over_verbose_correct if the reference is short. Model not downloaded."
                    
                return {"choices": [{"text": text}]}
                
        mock = MockModel()
        # DELIBERATELY NOT CACHING THE MOCK MODEL
        # This allows the API to automatically pick up the real weights as soon as the download finishes!
        return mock


@app.get("/")
def read_root():
    return FileResponse(str(static_dir / "index.html"))

@app.post("/api/evaluate")
async def evaluate_endpoint(req: GenerateRequest):
    if req.task_name not in TASKS:
        raise HTTPException(status_code=400, detail="Invalid task name")
        
    model = get_model(req.model_name)
    task_config = TASKS[req.task_name]
    prompt_template = getattr(prompts, task_config["prompt_template"])
    prompt = prompt_template.format(input=req.input_text)
    
    start = time.time()
    output = model(
        prompt, 
        max_tokens=GENERATION_CONFIG["max_new_tokens"],
        temperature=GENERATION_CONFIG["temperature"],
        top_p=GENERATION_CONFIG["top_p"],
        stop=GENERATION_CONFIG.get("stop", [])
    )
    latency = time.time() - start
    
    response_text = output["choices"][0]["text"].strip()
    
    # Evaluate correctness and taxonomy
    correct = False
    failure_type = "correct"
    
    if req.task_name == "qa":
        correct = evaluate_qa(response_text, req.reference_text)
        failure_type = classify_qa_failure(response_text, req.reference_text)
    elif req.task_name == "reasoning":
        correct = evaluate_reasoning(response_text, req.reference_text)
        failure_type = classify_reasoning_failure(response_text, req.reference_text)
    elif req.task_name == "summarization":
        correct = evaluate_summarization(response_text, req.reference_text)
        failure_type = classify_summarization_failure(response_text, req.reference_text)
        
    return {
        "output": response_text,
        "latency_sec": round(latency, 2),
        "correct": correct,
        "failure_type": failure_type
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
