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

openai.api_key = "api-key"
from data import making_prompt
from generation import create_generate_hf

parser = argparse.ArgumentParser(description = "Input")

parser.add_argument("model")
parser.add_argument("dataset")
parser.add_argument("--difficulty")

def Gpt1(prompt):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [
            {"role": "system", "content": "You are a skilled programmer."},
            {"role": "user", "content": prompt},
        ],
        temperature = 0.3
    )
    
    output =  response.choices[0].message.content
    return output

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
    
    
args = parser.parse_args()

dataset = args.dataset
model_name = args.model
diff = args.difficulty

if("human" in dataset):
    dataset = load_dataset("openai_humaneval", trust_remote_code=True, split="test")
    df = pd.DataFrame.from_records(dataset)
    if("Instruct" in model or "GPT" in model):
        prompts = making_prompt("HumanEval", df)
    else:
        prompts = making_prompt("HumanDirect", df)
else:
    dataset = load_dataset("codeparrot/apps", trust_remote_code=True, split="test")
    df = pd.DataFrame.from_records(dataset)
    if("Instruct" in model or "GPT" in model):
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

total_implementations = []
score_list = []
test_case = []
count = 0
stop_tokens = ["[/PYTHON]"]
for prompt in tqdm(prompts, desc="Implementation"):
    time_flag = False
    if count == 50:
        break
    example_implementations = []
    index = 0
    g_results = []
    current_time = datetime.now()
    while len(example_implementations) < 10:
        flag = False
        # Generate implementation
        completions = generate_fn([prompt[0]], stop_tokens)[0]
        # G_Prompt = "Do not print any examples" + prompt[0]
        # for j in range(2):
        #     completion = Gpt1(G_Prompt)
        #     completions.append(completion)
        # Extract implementation
        for text, _ in completions:
        # for text in completions:
            implementation, status = extract_implementation(text)
            # implementation, status = Gpt_extract(text)
            # implementation, status = extract_python(text)
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
        # time1 = datetime.now()
        # if(time1 - current_time > timedelta(minutes=5)):
        #     count -= 1
        #     time_flag = True
        #     break
    if(not time_flag):
        total_implementations.append(example_implementations)
        test_case.append(prompt[1])
    count +=1 
    all_scores = []
    for x in g_results:
        output = parse_output(x)
        if output == 0:
            continue
        all_scores.append(output)
    score = sum(all_scores) / len(all_scores)
    score_list.append(score)
    print(score)
    print("\n===================================================================================\n")
    

# print("\n===================================================================================\n")
# print(sum(score_list) / len(score_list))

# with open("7-P-3-C-scores", "w") as f:
#     for score in score_list:
#         f.write(f"{score}\n")
#     f.write("\n===================================================================================\n")
#     f.write(str(sum(score_list) / len(score_list)))
#     f.close()



# problem_num = 0

# for example_implementations in total_implementations:
#     count = 0
#     for implementation, status in example_implementations:
#         if implementation.find("def") == -1:
#             def_index = 10
#             name_index = 18
#         else:
#             def_index = implementation.index("def")
#             if implementation.find(":") == -1:
#                 name_index = def_index + 8
#             else:
#                 name_index = implementation.find(":")
#         name = implementation[def_index:name_index]
#         text = name.replace("_", "")
#         text = name.replace(" ", "")
#         code = delete_docstring(implementation)
#         with open(f"code-7-3-P-Apps-C-outputs/{problem_num}/{count}.py", "w") as f:
#             f.write(code)
#             f.close()
#         count += 1
#     problem_num += 1

# for i in range(164):
#     with open(f"code-7-3-P-Apps-C-testcase/{i}/{i}.txt", "w") as f:
#         f.write(test_case[i])
#         f.close()

        
