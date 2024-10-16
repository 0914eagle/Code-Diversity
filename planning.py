from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList)
import torch
import pandas as pd
import ast
from loguru import logger
from tqdm import tqdm
import openai
import re
from datasets import load_dataset
from datetime import datetime, timedelta

import argparse
from data import making_prompt
from generation import create_generate_hf

parser = argparse.ArgumentParser(desciption = "Input")

parser.add_argument("model")
parser.add_argument("dataset")
parser.add_argument("api_key")
parser.add_argument("--difficulty")
    
args = parser.parse_args()

openai.api_key = args.api_key

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

def self_reflect(prompt):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [
            {"role": "system", "content": "The self-reflection must cover every aspect of the problem. Pay attention to small details and nuances in the problem description."},
            {"role": "user", "content": prompt},
        ],
        temperature = 0.2
    )
    output =  response.choices[0].message.content
    return output

def sudo_sol(prompt):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [
            {"role": "system", "content": "Pay attention to small details and nuances in the problem description."},
            {"role": "user", "content": prompt},
        ],
        temperature = 0.2
    )
    output =  response.choices[0].message.content
    return output

def choose(prompt):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [
            {"role": "user", "content": prompt},
        ],
        temperature = 0.2
    )
    output =  response.choices[0].message.content
    return output

def Gpt2(prompt):
    response = openai.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "system", "content": "You are a skilled programmer."},
            {"role": "user", "content": prompt},
        ]
    )
    
    output =  response.choices[0].message.content
    return output


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

def delete_docstring(code: str) -> str:
    code = re.sub(r'"""(.|\n)*?"""', "", code)
    return code

def parse_output(output: str) -> float:
    output = output.strip()
    score = re.search(r"(\d+)", output)
    if score is None:
        print(output)
        return 0
    score = float(score.group(1))
    return score

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


total_implementations = []
score_list = []
individual_scores = []
test_case = []
count = 0
stop_tokens = ["[/PYTHON]"]
for prompt in tqdm(prompts, desc="Implementation"):
    
    if count == 50:
        break
    example_implementations = []
    index = 0
    g_results = []
    description = prompt[0]
    
    signature = prompt[2]
    
    with open("/home/0914eagle/CodeDiversity/self-reflect.txt", "r") as f:
        reflect = f.read()
        f.close()
    reflect = reflect.replace("{description}", description)
    reflection = self_reflect(reflect)
    
    with open("/home/0914eagle/CodeDiversity/generate_possible.txt", "r") as f:
        possibles = f.read()
        f.close()
    possibles = possibles.replace("{description}", description)
    possibles = possibles.replace("{self_reflection}", reflection)
    possible = sudo_sol(possibles)
    
    with open("/home/0914eagle/CodeDiversity/choose_good.txt", "r") as f:
        good = f.read()
        f.close()
    good = good.replace("{description}", description)
    good = good.replace("{self_reflection}", reflection)
    good = good.replace("{s_possible_solutions_str}", possible)
    goods = choose(good)
    
    with open("/home/0914eagle/CodeDiversity/App_prompt.txt", "r") as f:
        app_prompt = f.read()
        f.close()
    app_prompt = app_prompt.replace("{description}", description)
    app_prompt = app_prompt.replace("{self_reflection}", reflection)
    app_prompt = app_prompt.replace("{s_best_solution}", possible)
    app_prompt = app_prompt.replace("{signature}", signature)

    while len(example_implementations) < 10:
        flag = False
        # Generate implementation
        completions = []
        for j in range(2):
            completion = Gpt1(app_prompt)
            completions.append(completion)
        # Extract implementation
        for text in completions:
            implementation, status = Gpt_extract(text)
            if (len(implementation) < 10):
                flag = True
                if(len(example_implementations) % 2 == 1):
                    example_implementations.pop()
                break
            if(status != "ok"):
                flag = True
                if(len(example_implementations) % 2 == 1):
                    example_implementations.pop()
                break
            example_implementations.append((implementation, status))
        if not flag:
            code1 = delete_docstring(example_implementations[index][0])
            code2 = delete_docstring(example_implementations[index + 1][0])
            GPTprompt = open("/home/0914eagle/CodeDiversity/prompt.txt").read()
            GptCodePrompt = GPTprompt + "\nThe first code is, " +code1+ "\n And the second code is "+code2
            result = Gpt2(GptCodePrompt)
            g_results.append(result)
            index += 2
        
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
