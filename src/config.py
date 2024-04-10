import os

zip_url = "https://drive.google.com/file/d/150xHaqYwEy_H64VaF7kVOMOOdnG5cy-3/view?usp=sharing"

#necessary path variables
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/data"
patch_dataset_path = data_path + "/CBIS-DDSM-Patches"
TEM_dataset_path = data_path + "/CBIS-DDSM-TEM-Features"
patch_zip_path = data_path + "/CBIS-DDSM-Patches.zip"

saved_models_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/saved_models"
results_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/results"