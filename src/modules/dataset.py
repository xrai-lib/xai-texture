import os
import config
from preprocessing.TEM import generate_TEM_dataset
from preprocessing.patches import generate_patch_dataset
from utils import clear_screen

def check_patch_dataset():
    print("Checking for patch dataset")
    if 'CBIS-DDSM-Patches' in os.listdir(config.data_path) and len(os.listdir(config.patch_dataset_path)) > 0:
        print("CBIS-DDSM-Patches dataset located successfully")
        return True
    else:
        print("The repository does not contain the CBIS-DDSM-Patches dataset.")
        return False
    
def check_TEM_dataset():
    print("Checking for TEM dataset")
    if 'CBIS-DDSM-TEM-Features' in os.listdir(config.data_path) and len(os.listdir(config.TEM_dataset_path)) > 0:
        print("CBIS-DDSM-TEM-Features dataset located successfully")
        return True
    else:
        print("The repository does not contain the CBIS-DDSM-TEM-Features dataset.")
        return False

def dataset_module():
    clear_screen() #clears the terminal screen
    
    #ensure dataset presence
    if not check_patch_dataset():
        generate_patch_dataset()

    if not check_TEM_dataset():
        generate_TEM_dataset(True)
        generate_TEM_dataset(False)

    return