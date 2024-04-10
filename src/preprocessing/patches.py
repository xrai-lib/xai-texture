import config
import zipfile
import subprocess

def download_file_wget(url, local_filename=None):
    # Prepare the wget command
    wget_command = ['wget', url]
    
    # If a local filename is provided, add output document argument
    if local_filename:
        wget_command += ['-O', local_filename]
    
    # Execute the wget command
    result = subprocess.run(wget_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check if the command executed successfully
    if result.returncode == 0:
        print("Download successful")
    else:
        print(f"Error downloading file: {result.stderr}")

def generate_patch_dataset():
    print("Downloading CBIS-DDSM-Patches Dataset")
    url = config.zip_url
    download_file_wget(url, config.patch_zip_path)
    
    print("Extracting CBIS-DDSM-Patches Dataset")
    # Path to the ZIP file
    zip_file_path = config.patch_zip_path
    # Destination directory where the contents will be extracted
    destination_directory = config.data_path

    # Open the ZIP file in read mode
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the destination directory
        zip_ref.extractall(destination_directory)