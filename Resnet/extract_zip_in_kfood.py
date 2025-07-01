import os
import zipfile

# Define the path to the kfood directory
kfood_dir = os.path.join("한국 음식 이미지", "kfood")

# List all files and directories in kfood_dir
for item in os.listdir(kfood_dir):
    if item.endswith(".zip"):
        zip_path = os.path.join(kfood_dir, item)
        folder_name = item[:-4]  # Remove .zip extension
        folder_path = os.path.join(kfood_dir, folder_name)
        # Check if the folder already exists
        if not os.path.exists(folder_path):
            print(f"Extracting {zip_path} to {folder_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(folder_path)
        else:
            print(f"Folder {folder_path} already exists. Skipping extraction.")
