from typing import Callable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList)
import os
import subprocess
import torch
from loguru import logger


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        self.check_fn = lambda decoded_generation: any([
            stop_string in decoded_generation
            for stop_string in self.eof_strings
        ])

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:])
        return all([
            self.check_fn(decoded_generation)
            for decoded_generation in decoded_generations
        ])
        
def create_tokenizer(checkpoint_name: str):
    params = {
        "pretrained_model_name_or_path": checkpoint_name,
        "trust_remote_code": True,
        "cache_dir" : "/share0/0914eagle/",
        "device_map": "auto",
    }
    
    if checkpoint_name == "bigcode/santacoder":
        params["pad_token"] = "<|endoftext|>"
    return AutoTokenizer.from_pretrained(**params)

def create_model(cache_dir: str, checkpoint_name:str):
    params = {
        "pretrained_model_name_or_path": checkpoint_name,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "cache_dir": "/share0/0914eagle/",
        "max_length": 1024,
        "device_map": "auto",
    }
    # if checkpoint_name.startswith("codellama/CodeLlama"):
    #     params["use_flash_attention_2"] = True
    return AutoModelForCausalLM.from_pretrained(**params)

def create_generate_hf(checkpoint_name: str) -> Callable:
    tokenizer = create_tokenizer(checkpoint_name)

    # Get pad token for batch generation
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    # NOTE: We use eos token as pad token if pad token is not available
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have a pad token or an eos token")
        tokenizer.pad_token = tokenizer.bos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("/share0/0914eagle/", checkpoint_name).eval()
    
    def generate_fn(
        prompt: list[str],
        stop_sequence: list[str]
    ) -> list[list[tuple[str, str]]]:
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=4,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_k=None,
            top_p=0.95,
        )
        with torch.inference_mode():
            encodings = tokenizer(prompt, padding=True, truncation=True, return_token_type_ids=False, return_tensors="pt", max_length=4098)
            encodings = encodings.to(device)
            stopping_criteria = StoppingCriteriaList([
                EndOfFunctionCriteria(
                        start_length=encodings["input_ids"].shape[1],
                        eof_strings=stop_sequence,
                        tokenizer=tokenizer,
                    )
                ])
            output_ids = model.generate(
                    **encodings,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                )
            output_ids = output_ids.tolist()
            completions = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
        def determine_finish_reason(text: str, output_id: list[int]):
            if any(
                    text.endswith(stop_sequence)
                    for stop_sequence in stop_sequence):
                return "stop_sequence"
            elif tokenizer.eos_token_id in output_id:
                return "eos_token"
            else:
                return "length"

        completions = [(text, determine_finish_reason(text, output_id))
                       for text, output_id in zip(completions, output_ids)]
        completions = [
            completions[idx:idx + 2]
            for idx in range(0,
                             len(prompt) *
                             2, 2)
        ]
        return completions
    
    return generate_fn