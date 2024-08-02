import torch
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer
from kagglehub import model_download
from time import time

def check_cuda():
    return cuda.is_available()

def download_model(model_name):
    base_cache_dir = os.path.expanduser("~/.cache/kagglehub/models")
    model_path = os.path.join(base_cache_dir, model_name.replace("/", "_"))

    if os.path.exists(model_path):
        print("Model already downloaded.")
    else:
        print("Downloading model...")
        model_path = model_download(model_name)
        print("Path to model files:", model_path)

    return model_path

def setup_model(model_id, device):
    # Quantization configuration
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    time_1 = time()
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    time_2 = time()
    print(f"Prepare model, tokenizer: {round(time_2 - time_1, 3)} sec.")
    return model, tokenizer

def setup_pipeline(model, tokenizer):
    time_1 = time()
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    time_2 = time()
    print(f"Prepare pipeline: {round(time_2 - time_1, 3)} sec.")
    return query_pipeline

def test_model(tokenizer, pipeline, prompt_to_test):
    time_1 = time()
    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    time_2 = time()
    print(f"Test inference: {round(time_2 - time_1, 3)} sec.")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")