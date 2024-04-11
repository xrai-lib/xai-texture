import os

zip_url = "https://zenodo.org/records/10960991/files/CBIS-DDSM-Patches.zip?download=1"

#necessary path variables
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/data"
patch_dataset_path = data_path + "/CBIS-DDSM-Patches"
TEM_dataset_path = data_path + "/CBIS-DDSM-TEM-Features"
patch_zip_path = data_path + "/CBIS-DDSM-Patches.zip"

saved_models_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/saved_models"
results_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/results"