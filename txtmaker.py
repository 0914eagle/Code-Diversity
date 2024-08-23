import argparse


for i in range(5):
    with open(f'input_{i}.txt', 'w') as file:
    # Write the zip file paths
        for j in range(50):
            file.write(f"folder_path/{j}/{i}.zip\n")
            
argparser = argparse.ArgumentParser()
argparser.add_argument("input_path", type=str, required=True)
args = argparser.parse_args()

# The input file which contains the list of files to be zipped
folder_path = args.input_path