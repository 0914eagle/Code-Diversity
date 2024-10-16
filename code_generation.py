from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList)
import torch
import pandas as pd
import ast
from tqdm import tqdm
import openai
import re
from datasets import load_dataset
from datetime import datetime, timedelta
import argparse


from data import making_prompt
from generation import create_generate_hf

parser = argparse.ArgumentParser(description = "Input")

parser.add_argument("model")
parser.add_argument("dataset")
parser.add_argument("api_key")
parser.add_argument("--difficulty")
    
args = parser.parse_args()

openai.api_key = args.api_key

# GPT Inference for Code Generation
def Gpt1(prompt):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [
            {"role": "system", "content": "You are a skilled programmer."},
            {"role": "user", "content": prompt},
        ],
        temperature = 0.2
    )
    
    output =  response.choices[0].message.content
    return output

# GPT Inference for Code Similarity
def Gpt2(prompt):
    response = openai.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {"role": "system", "content": "You are a skilled programmer."},
            {"role": "user", "content": prompt},
        ]
    )
    
    output =  response.choices[0].message.content
    return output

def extract_implementation_from_turn(text: str) -> tuple[str, str]:
    try:
        index = text.index("[/INST]") + len("[/INST]")
        index_start = text.index("[PYTHON]", index) + len("[PYTHON]")
    except ValueError:
        return text, "[/INST] or [PYTHON] tag not found"

    try:
        index_end = text.index("[/PYTHON]", index)
        text = text[index_start:index_end]
        return text, "ok"
    except ValueError:
        text = text[index_start:]
        return text, "[/PYTHON] tag not found"

# Extract implementation for Instruction tuned models
def extract_implementation(text: str) -> tuple[str, str]:
    text = text.strip()
    implementation = ""
    index_start = 0
    if(text.startswith("<s>")):
        index_start = text.index("<s>") + len("<s>") + 1
    while True:
        try:
            # Find next instruction
            index_end = text.index("[INST]", index_start + 1)
            turn = text[index_start:index_end]
            implementation_chunk, status = extract_implementation_from_turn(turn)
            implementation += f"{implementation_chunk}\n"
            index_start = index_end
            if status != "ok":
                return implementation, status
        except ValueError:
            turn = text[index_start:]
            implementation_chunk, status = extract_implementation_from_turn(turn)
            implementation += f"{implementation_chunk}\n"
            return implementation, status
        
# Extract implementation for base models
def extract_direct(text: str):
    text = text.strip()
    try:
        index_start = text.index("```python") + len("```python")
        index_end = text.index("```", index_start)
        text = text[index_start:index_end]
        return text, "ok"
    except ValueError:
        return text, "```python tag not found"
    
def delete_docstring(code: str) -> str:
    code = re.sub(r'"""(.|\n)*?"""', "", code)
    return code

# Find the code similarity score from the output
def parse_output(output: str) -> float:
    output = output.strip()
    score = re.search(r"(\d+)", output)
    if score is None:
        print(output)
        return 0
    score = float(score.group(1))
    return score

def remove_after_stop_token(completion: str, prompt: str,
                            stop_tokens: list[str]) -> str:
    min_stop_index = len(completion)
    for stop_token in stop_tokens:
        stop_index = completion.find(stop_token, len(prompt))
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return completion[:min_stop_index]

# Extract implementation for GPT 3.5 model
def Gpt_extract(text: str) -> tuple[str,str]:
    text = text.strip()
    implementation = ""
    try:
        index = text.index("[PYTHON]") + len("[PYTHON]")
        index_end = text.index("[/PYTHON]", index)
        implementation = text[index:index_end]
        
        return implementation, "ok"
    except ValueError:
        index = text.find("```python") + len("```python")
        if(index == -1):
            raise ValueError("Not Code")
        index_end = text.find("[/PYTHON]", index)
        if (index_end == -1):
            index_end = text.find("```", index+1)
        if (text.find("```", index+1)  < index_end):
            index_end = text.find("```", index+1)
        implementation = text[index:index_end]
        return implementation, "ok"
    
def extract_python(text: str) -> str:
    try:
        index_start = text.index("[PYTHON]") + len("[PYTHON]")
        if text.find("[/PYTHON]") == -1:
            index_end = text.index("[PYTHON]", index_start)
        else:
            index_end = text.index("[/PYTHON]", index_start)
        text = text[index_start:index_end]
        return text, "ok"
    except ValueError:
        return text, "error"

def parse_out(output: str) -> float:
    index = output.find("def")
    next = output.find("def", index + 1)
    if next == -1:
        return output, "ok"
    else:
        return output[:next], "ok"

dataset = args.dataset
model_name = args.model
diff = args.difficulty

if("human" in dataset):
    dataset = load_dataset("openai_humaneval", trust_remote_code=True, split="test")
    df = pd.DataFrame.from_records(dataset)
    if("Instruct" in model_name or "GPT" in model_name):
        prompts = making_prompt("HumanEval", df)
    else:
        prompts = making_prompt("HumanDirect", df)
else:
    dataset = load_dataset("codeparrot/apps", trust_remote_code=True, split="test")
    df = pd.DataFrame.from_records(dataset)
    if("Instruct" in model_name or "GPT" in model_name):
        Com_prompt, Itv_prompt, Itd_prompt = making_prompt("APPs", df)
        if("Com" in diff):
            prompts = Com_prompt
        elif ("Itv" in diff):
            prompts = Itv_prompt
        else:
            prompts = Itd_prompt
    else:
        D_Com_prompt, D_Itv_prompt, D_Itd_prompt = making_prompt("APPsDirect", df)
        if("Com" in diff):
            prompts = D_Com_prompt
        elif ("Itv" in diff):
            prompts = D_Itv_prompt
        else:
            prompts = D_Itd_prompt


generate_fn = create_generate_hf(model_name)

# Check the model type
GPT_flag = False
Instruct_flag = False
Direct_flag = False

if "GPT" in model_name:
    GPT_flag = True
elif "Instruct" in model_name:
    Instruct_flag = True
else:
    Direct_flag = True

total_implementations = []
score_list = []
individual_scores = []
test_case = []
count = 0
stop_tokens = ["[/PYTHON]"]

for prompt in tqdm(prompts, desc="Implementation"):
    
    # We sample 50 problems per datasetz
    if count == 50:
        break
    example_implementations = []
    index = 0
    g_results = []
    
    # We generate 10 implementations per problem
    while len(example_implementations) < 10:
        flag = False
        # Generate implementation
        
        # If model is GPT
        if GPT_flag:
            G_Prompt = "Do not print any examples" + prompt[0]
            for j in range(2):
                completion = Gpt1(G_Prompt)
                completions.append(completion)
                
            for text in completions:
                implementation, status = Gpt_extract(text)
        else:
            completions = generate_fn([prompt[0]], stop_tokens)[0]
        # Extract implementation
            for text, _ in completions:
                # If model is Instruction tuned
                if Instruct_flag:
                    implementation, status = extract_implementation(text)
                
                # If model is base model
                else:
                    implementation, status = extract_direct(text)
        
        # If implementation is not code, skip and pop the last implementation if odd
        if (len(implementation) < 10):
                flag = True
                if(len(example_implementations) % 2 == 1):
                    example_implementations.pop()
                break
        
        # If status is not ok, skip and pop the last implementation if odd
        if(status != "ok"):
                flag = True
                if(len(example_implementations) % 2 == 1):
                    example_implementations.pop()
                break
            
        example_implementations.append((implementation, status))
        
        # If the pair of implementations are generated well, we compare the reasoning-based similarity score between the two codes.
        if not flag:
            code1 = delete_docstring(example_implementations[index][0])
            code2 = delete_docstring(example_implementations[index + 1][0])
            GPTprompt = open("prompt.txt").read()
            GptCodePrompt = GPTprompt + "\nThe first code is, " +code1+ "\n And the second code is "+code2
            result = Gpt2(GptCodePrompt)
            g_results.append(result)
            index += 2
    
    # Save the results
    total_implementations.append(example_implementations)
    test_case.append(prompt[1])
    count +=1 
    all_scores = []
    for x in g_results:
        output = parse_output(x)
        if output == 0:
            continue
        all_scores.append(output)
        individual_scores.append(output)
    score = sum(all_scores) / len(all_scores)
    score_list.append(score)
    
# Save the similarity score
with open("reasoning_based_score", "w") as f:
    for score in individual_scores:
        f.write(f"{score}\n")
        f.write("\n===================================================================================\n")
    for score in score_list:
        f.write(f"{score}\n")
    f.write("\n===================================================================================\n")
    f.write(str(sum(score_list) / len(score_list)))
    f.close()

problem_num = 0

for example_implementations in total_implementations:
    count = 0
    for implementation, status in example_implementations:
        code = delete_docstring(implementation)
        with open(f"{model_name}/{problem_num}/{count}.py", "w") as f:
            f.write(code)
            f.close()
        count += 1
    problem_num += 1


for i in range(len(test_case)):
    with open(f"{model_name}-testcase/{i}/{i}.txt", "w") as f:
        f.write(test_case[i])
        f.close()
