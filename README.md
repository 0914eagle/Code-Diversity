# Is Functional Correctness Enough to Evaluate Code Language Models?Exploring Diversity of Generated Codes

### Appendix

In AAAI25_CodeLM_Diversity_Appendix, you can find the appendix of the paper "Is Functional Correctness Enough to Evaluate Code Language Models? Exploring Diversity of Generated Codes".

## Code

This Folder contains the codes used in the experiments of the paper "Is Functional Correctness Enough to Evaluate Code Language Models? Exploring Diversity of Generated Codes".

## Requirements

In requirements.txt you can find the necessary libraries to run the code.

## Code Generation and Reasoning-based Similarity Evaluation

Before generate codes, you need to run the following command to build the necessary folders:

`python filemaker.py model_name <model_name to generate>`

To generate codes, you can use the following command:

`python code_generation.py model <model_name> dataset <HumanEval/APPs> api_key <your openai api_key> --difficulty <In APPs>`

Then you will get the folder named "**model_name/num**"and you can find the generated codes in the "**num**" folder.

And the Reasoning-based similarity scores are going to be saved in the "**reasoning_based_score**".

## Embedding-Based Code Similiarity Evaluation

To evaluate the similarity between generated codes by Embedding-based methods, you can use the following command:

`pyhton codeBert.py input_file <path_to_input_file>`

The input_file is the path to the generated codes.

## Token-Based Code Similarity Evaluation

To evaluate the similarity between generated codes by Token-based methods,
you run **zipmaker.py** for making zip files. You can use the following command:

`python zipmaker.py input_file <path_to_input_file> output_file <path_to_output_file>`

Then, you can run **txtmaker.py** for preparing the input files for the token similarity evaluation. You can use the following command:

`python txtmaker.py input_file <path_to_input_file>`

Finally, you can follow the README.md in
[SourcererCC](https://github.com/Mondego/SourcererCC).

## Planning for Code Generation

To apply planning prompt during code generation, you can use the following command:

`python planning.py model <Only GPT available> dataset <HumanEval/APPs> api_key <your openai api_key> --difficulty <In APPs>`

Then you will get the folder named "**model_name/num**"and you can find the generated codes in the "**num**" folder.
