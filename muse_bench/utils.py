import json
import pandas as pd
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def read_json(fpath: str) -> Dict | List:
    with open(fpath, 'r') as f:
        return json.load(f)


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()


def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)


def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return f.write(obj)


def write_csv(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


def load_model(
    model_dir: str,
    quantize_4bit: int = 0,
    quantize_8bit: int = 0,
    **kwargs
):
    if quantize_4bit == 1:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map='auto',
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            **kwargs
        )
    if quantize_8bit == 1:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map='auto',
            quantization_config=bnb_config,
            **kwargs
        )
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        **kwargs
    )


def load_tokenizer(tokenizer_dir: str, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
    
