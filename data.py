import ast
import random
from functools import partial
from datasets import load_dataset

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

def parse_signature(code: str) -> str:
    node = ast.parse(code)
    for segment in node.body:
       if isinstance(segment, ast.FunctionDef):
            # Remove docstring
            segment.body = []
            # Remove "def "
            signature = ast.unparse(segment)[4:]
            return signature
    raise ValueError("Function signature not found")


def parse_docstring(code: str, clean: bool=False) -> str:
    node = ast.parse(code)
    for segment in node.body:
        if isinstance(segment, ast.FunctionDef):
            docstring = ast.get_docstring(segment, clean=clean)
            return docstring
    raise ValueError("Function signature not found")

def direct_humaneval(row: dict) -> str:
    prompt = row["prompt"].strip()
    # prompt = row["prompt"]
    # signature = parse_signature(row["prompt"])
    # docstring = parse_docstring(row["prompt"])
    # prompt = f"```python\n{signature}"
    test_case = row["test"]
    out = [prompt, test_case]
    return out

def create_prompt(row: dict) -> list:
    prompt = row["prompt"]
    test_case = row["test"]
    signature = parse_signature(row["prompt"])
    docstring = parse_docstring(row["prompt"])
    prompt = f"<s> [INST] Write a Python function `{signature}` to solve the following problem:\n{docstring}\n\n[/INST]\n{prompt} \n ```python\n def {signature}"
    out = [prompt, test_case]
    return out


def create_APPs_prompt(row: dict) -> str:
    prompt = row["question"]
    prompt = f"<s> [INST] Write a Python code to solve the following problem and there must include data inputer, no solely one function.:Your code should start with a [PYTHON] tag and end with a [/PYTHON] tag. The code output must follow this structure:def ~(...):...return ...def ~(...):...return ... ... if __name__ == '__main__':...\n{prompt}\n\n[/INST]\n[PYTHON]\n"
    test_case = row["input_output"]
    out = [prompt, test_case]
    return out

def direct_Apps(row: dict) -> str:
    # prompt = row["question"].strip()
    prompt = row["question"]
    prompt = f"The code output must follow this structure:def ~(...):...return ...def ~(...):...return ... ... if __name__ == '__main__':...\n{prompt} \n ```python\n"
    test_case = row["input_output"]
    out = [prompt, test_case]
    return out

def making_prompt(name: str, df: pd.DataFrame):
    if(name == "HumanEval"):
        
        prompts = df.apply(create_prompt, axis=1)
        prompts = prompts.sample(frac = 1)
        return prompts
    elif (name == "APPs"):
        
        com_df = df[df["difficulty"] == "competition"]
        itv_df = df[df["difficulty"] == "interview"]
        itd_df = df[df["difficulty"] == "introductory"]
        
        # Random sampling
        prompts = com_df.apply(create_APPs_prompt, axis=1)
        com_prompts = prompts.sample(frac = 1)
        
        prompts = itv_df.apply(create_APPs_prompt, axis=1)
        itv_prompts = prompts.sample(frac = 1)
        
        prompts = itd_df.apply(create_APPs_prompt, axis=1)
        itd_prompts = prompts.sample(frac = 1)
        
        return com_prompts, itv_prompts, itd_prompts
    elif(name == "HumanDirect"):
        prompts = df.apply(direct_humaneval, axis=1)
        prompts = prompts.sample(frac = 1)
        return prompts
    elif(name == "APPsDirect"):
        com_df = df[df["difficulty"] == "competition"]
        itv_df = df[df["difficulty"] == "interview"]
        itd_df = df[df["difficulty"] == "introductory"]
        
        # Random sampling
        prompts = com_df.apply(direct_Apps, axis=1)
        com_prompts = prompts.sample(frac = 1)
        
        prompts = itv_df.apply(direct_Apps, axis=1)
        itv_prompts = prompts.sample(frac = 1)
        
        prompts = itd_df.apply(direct_Apps, axis=1)
        itd_prompts = prompts.sample(frac = 1)
        
        return com_prompts, itv_prompts, itd_prompts
    
    
    
        