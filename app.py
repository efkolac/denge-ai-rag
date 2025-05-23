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
MODEL_REPO = "Qwen/Qwen3-14B"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
LOCAL_MODEL_DIR = MODEL_DIR / "Qwen/Qwen3-14B"
hf = "hf_SAzoqialcumI"
hf += "kbbCplrGbgwBandoXVnTUt"

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

        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf
        )
                
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


def handler(event):
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
        
        # Format prompt
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype="auto",
            device_map="auto"
        )

        # prepare the model input
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        try:
    # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)
        # conduct text completion
        # generated_ids = model.generate(
        #     **model_inputs,
        #     max_new_tokens=32768
        # )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # if context:
        #     formatted_prompt = f"<s>[INST]  <context>\n{context}\n</context>\n\n{prompt} [/INST]"
        # else:
        #     formatted_prompt = f"<s>[INST]  {prompt} [/INST]"
        # text = tokenizer.apply_chat_template(
        #     formatted_prompt,
        #     tokenize=False,
        #     add_generation_prompt=True,
        #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        # )
        # Generate response
        # inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **model_inputs,
        #         max_length=max_length,
        #         temperature=temperature,
        #         top_p=top_p,
        #         do_sample=True
        #     )
        
        # # Decode and clean response
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response = response.split("[/INST]")[-1].strip()
        
        return {
            "response": content
            # "sources": sources  # Include the source files in the response
        }
    
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