from sentence_transformers import SentenceTransformer, util
import os
import argparse

#This list the defines the different programm codes


model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

cossim = []

argparser = argparse.ArgumentParser()
argparser.add_argument("path", type=str, required=True)
args = argparser.parse_args()

folder_path = args.path

for i in range(1, 50):
    # Calculate the similarity between the generated codes.
    for j in range(0, 10, 2):
        with open(f"{folder_path}/{i}/{j}.py", "r", encoding='UTF8') as code1:
            code1 = code1.read()
        with open(f"{folder_path}/{i}/{j+1}.py", "r", encoding='UTF8') as code2:
            code2 = code2.read()
    # First Code
    code_emb = model.encode(code1, convert_to_tensor=True)
    # Second Code
    query_emb = model.encode(code2, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, code_emb)[0]
    for top in hits:
        cossim.append(top['score'])
        
score = sum(cossim) / len(cossim)

print(f"Average cosine similarity: {score}")
