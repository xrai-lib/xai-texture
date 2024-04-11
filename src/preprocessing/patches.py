import config
import zipfile
import requests

def download_file(url, save_path):
    """
    Download a file from a specified URL and save it locally.

    Args:
    url (str): URL of the file to download.
    save_path (str): Local path to save the file.
    """
    # Send a GET request to the URL
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check if the request was successful
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):  # Download the file in chunks
                f.write(chunk)

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