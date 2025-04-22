import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from pathlib import Path
import logging
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Model configuration
MODEL_REPO = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
LOCAL_MODEL_DIR = MODEL_DIR / "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = None
model = None

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def download_model():
    """Download the model with retries and progress tracking"""
    try:
        # Create fresh directory
        if LOCAL_MODEL_DIR.exists():
            shutil.rmtree(LOCAL_MODEL_DIR)
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {MODEL_REPO} to {LOCAL_MODEL_DIR}...")

        # Use snapshot_download for more reliable downloads
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin", "*.txt"],
            ignore_patterns=["*.h5", "*.ot", "*.tflite"],
            max_workers=4
        )

        logger.info("Download complete!")
        return LOCAL_MODEL_DIR

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

def load_model():
    """Load model and tokenizer with error handling"""
    global tokenizer, model
    
    try:
        # Download model if not already present
        if not (LOCAL_MODEL_DIR / "model.safetensors").exists():
            download_model()

        logger.info("Loading model from local directory...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            trust_remote_code=True
        )
        
        # Load model with device map and memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token= hf_token
        )
        
        logger.info(f"Model loaded on {model.device}")
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Load model when the container starts
load_model()

def handler(event):
    """
    RunPod serverless handler function
    Expected input format:
    {
        "input": {
            "prompt": "Your question here",
            "context": "Optional context",
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    """
    try:
        input_data = event.get('input', {})
        
        # Validate input
        if 'prompt' not in input_data:
            return {"error": "Missing required field 'prompt' in input"}
        
        # Get generation parameters with defaults
        prompt = input_data['prompt']
        context = input_data.get('context')
        max_length = input_data.get('max_length', 2048)
        temperature = input_data.get('temperature', 0.7)
        top_p = input_data.get('top_p', 0.9)
        
        folder_path = input_data.get('folder_path', './context_files')  # Default folder
        filecount=0
        # Read and append files from folder if it exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            file_contents = []
            for file_name in sorted(os.listdir(folder_path)):
                file_path = Path(folder_path) / file_name
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            filecount+=1
                            file_contents.append(f"File: {file_name}\n{f.read().strip()}")
                    except Exception as e:
                        logger.warning(f"Could not read file {file_name}: {str(e)}")
            
            if file_contents:
                files_context = "\n\n".join(file_contents)
                context = f"{context}\n\n{files_context}" if context else files_context
            else:
                return {"response":"no context."}
        else:
            return {"response":"Böyle bir dosya bulunamadı."}
        # Format prompt
        if context:
            formatted_prompt = f"<s>[INST]  <context>\n{context}\n</context>\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST]  {prompt} [/INST]"
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        
        return {"response": f"{filecount} {response}"}
    
    except torch.cuda.OutOfMemoryError:
        return {"error": "GPU out of memory - try reducing max_length"}
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return {"error": str(e)}

if __name__ == '__main__':
    # test_event = {
    #     "input": {
    #         "prompt": "Explain quantum computing in simple terms",
    #         "max_length": 200,
    #         "temperature": 0.7
    #     }
    # }
    
    # # Simulate RunPod's call
    # result = handler(test_event)
    # print("Test Output:", result)
    runpod.serverless.start({
        "handler": handler
    })