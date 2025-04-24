import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from pathlib import Path
import logging
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Model configuration
MODEL_REPO = "deepseek-ai/deepseek-moe-16b-chat"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
LOCAL_MODEL_DIR = MODEL_DIR / "deepseek-ai/deepseek-moe-16b-chat"
hf = "hf_hOpJaCBfzEQSo"
hf += "EEtQDTotIwMxKOFeVZEVL"

tokenizer = None
model = None

load_dotenv()

def download_model():
    """Download the model with retries"""
    try:
        if LOCAL_MODEL_DIR.exists():
            shutil.rmtree(LOCAL_MODEL_DIR)
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {MODEL_REPO}...")
        
        # Download essential files
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]

        
        for file in required_files:
            try:
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=file,
                    local_dir=LOCAL_MODEL_DIR,
                    resume_download=True,
                    token=hf
                )
            except Exception as e:
                logger.warning(f"Couldn't download {file}: {str(e)}")
                
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
            token= hf
        )
        
        logger.info(f"Model loaded on {model.device}")
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Load model when the container starts
load_model()

def get_relevant_context(prompt, folder_path, top_k=3):
    context_chunks = []
    file_data = []

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_data.append((file_name, content))

    if not file_data:
        return ""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2",
                                  token=hf)
    # Create embeddings
    prompt_embedding = embed_model.encode(prompt, convert_to_tensor=True)
    passages = [content for _, content in file_data]
    passage_embeddings = embed_model.encode(passages, convert_to_tensor=True)

    # Semantic similarity
    scores = util.cos_sim(prompt_embedding, passage_embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(file_data)))

    for idx in top_results.indices:
        name, text = file_data[idx]
        context_chunks.append(f"File: {name}\n{text.strip()}")

    return "\n\n".join(context_chunks)

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
            files_context = get_relevant_context(prompt, folder_path)
            if files_context:
                context = f"{context}\n\n{files_context}" if context else files_context
            else:
                return {"response": "No relevant context found."}
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
        
        return {"response": f"{hf} {response}"}
    
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