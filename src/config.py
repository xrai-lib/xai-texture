import os

zip_url = "https://zenodo.org/records/10960991/files/CBIS-DDSM-Patches.zip?download=1"

#necessary path variables
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/data"
patch_dataset_path = data_path + "/CBIS-DDSM-Patches"
TEM_dataset_path = data_path + "/CBIS-DDSM-TEM-Features"
patch_zip_path = data_path + "/CBIS-DDSM-Patches.zip"

maskdataset_path_mmsegmentation = "../data/SegmentationClass"
unet_config_file_path = "models/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py"

saved_models_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/saved_models"
results_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/results"