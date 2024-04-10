import config
import zipfile

def generate_patch_dataset():
    print("Extracting CBIS-DDSM-Patches Dataset")
    # Path to the ZIP file
    zip_file_path = config.patch_zip_path
    # Destination directory where the contents will be extracted
    destination_directory = config.data_path

    # Open the ZIP file in read mode
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the destination directory
        zip_ref.extractall(destination_directory)