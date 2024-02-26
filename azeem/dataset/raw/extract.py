import os
import zipfile
import shutil
from tqdm import tqdm

def unzip_and_move(zip_file, target_dir):
    """
    Function to unzip a zip file and move its contents to a target directory.

    Args:
        zip_file (str): Path to the zip file.
        target_dir (str): Path to the target directory.

    Returns:
        None
    """
    # Open the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get the list of files in the zip file
        file_list = zip_ref.namelist()
        # Iterate over each file in the zip file
        for file in tqdm(file_list, desc=f'Extracting {zip_file}', unit='files'):
            # Extract the file to the target directory
            zip_ref.extract(file, target_dir)

def main():
    """
    Main function to unzip and move files.
    """
    # Create a directory to store extracted CSV files if it doesn't exist
    processed_dir = './dataset/processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Unzip train.zip
    train_zip = './dataset/raw/train.zip'
    unzip_and_move(train_zip, processed_dir)

    # Unzip test.zip
    test_zip = './dataset/raw/test.zip'
    unzip_and_move(test_zip, processed_dir)

    print("Extraction and move completed successfully.")

if __name__ == "__main__":
    main()
