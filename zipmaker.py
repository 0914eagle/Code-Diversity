import os
import zipfile
import argparse

def zip_files(files, folder_path, output_path, zip_index):
    zip_file_path = os.path.join(output_path, f'{zip_index}.zip')
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_path = os.path.join(folder_path, file)
            arcname = os.path.relpath(file_path, folder_path)
            zipf.write(file_path, arcname)

def zip_directory_in_batches(folder_path, output_path):
    for i in range(0, 50):
        batch_files = []
        folder_to_zip = f"{folder_path}/{i}"
        output_folder = f"{output_path}/{i}"
        os.makedirs(output_folder, exist_ok=True)
        files = os.listdir(folder_to_zip)
        batch_size = 2
        if(len(files) == 0):
            continue
        for j in range(0, 10, batch_size):
            batch_files = files[j:j + batch_size]
            zip_files(batch_files, folder_to_zip, output_folder, j // batch_size)

argparser = argparse.ArgumentParser()
argparser.add_argument("input_path", type=str, required=True)
argparser.add_argument("output_path", type=str, required=True)
args = argparser.parse_args()


# input path
folder_path = args.input_path

# output path
output_path = args.output_path

zip_directory_in_batches(folder_path, output_path)