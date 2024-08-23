import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('model_name', help='The name of the model to process')
args = argparser.parse_args()

os.mkdir(args.model_name)

for i in range(50):
    os.mkdir(f"{args.model_name}/{i}")