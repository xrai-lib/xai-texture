import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from utils import clear_screen
from models.fcn import train_fcn
from models.deeplabv3 import train_deeplab
from models.unet import train_unet
from models.hrnet import train_hrnet
from models.fpn import train_fpn
from models.linknet import train_linknet

import config
import cv2
import numpy as np


def prompt_model():
    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")
    print("4. HR-Net")
    print("5. FPN-Net")
    print("6. Link-Net")

    choice = None
    while True:
            try:
                choice = int(input("Select Model (1-6): "))
                if 1 <= choice <= 6:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 6 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_dataset():

    print("1. CBIS_DDSM")
    print("2. CBIS_DDSM_CLAHE")
    print("3. HAM10000")
    print("4. HAM10000_CLAHE")
    print("5. POLYP")
    print("6. POLYP_CLAHE")
    

    choice = None
    while True:
            try:
                choice = int(input("Select Dataset (1-6): "))
                if 1 <= choice <= 6:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 6 available datasets.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_feature_dataset():

    print("1. Feature 1 (L5E5 / E5L5)")
    print("2. Feature 2 (L5S5 / S5L5)")
    print("3. Feature 3 (L5R5 / L5R5)")
    print("4. Feature 4 (E5S5 / S5E5)")
    print("5. Feature 5 (E5R5 / R5E5)")
    print("6. Feature 6 (R5S5 / S5R5)")
    print("7. Feature 7 (S5S5)")
    print("8. Feature 8 (E5E5)")
    print("9. Feature 9 (R5R5)")
    print("10. Original")

    choice = None
    while True:
            try:
                choice = int(input("Select Dataset (1-10): "))
                if 1 <= choice <= 10:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 10 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

# Custom transform for resizing images
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resize(img, self.size)


class CancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name.split('.')[0] + '.png')
        
        # Read image and mask using cv2
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image loading was successful
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        # Check if mask loading was successful
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        
        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert mask to binary
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Convert images to PIL format
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        # Apply transforms if provided
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

# class CancerDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.image_transform = image_transform
#         self.mask_transform = mask_transform
#         self.images = os.listdir(images_dir)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_name = self.images[idx]
#         image_path = os.path.join(self.images_dir, image_name)
#         mask_path = os.path.join(self.masks_dir, image_name.split('.')[0] + '.png')
#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")
        
#         if self.image_transform:
#             image = self.image_transform(image)
        
#         if self.mask_transform:
#             mask = self.mask_transform(mask)
        
#         return image, mask


def create_data_loader(dataset_choice, feature_dataset_choice):
    
    image_transform = transforms.Compose([
        ResizeTransform((512, 512)),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        ResizeTransform((512, 512)),  # Resize to 256x256
        transforms.ToTensor()
    ])

    # Create your datasets and data loaders
    images_dir = ''
    if dataset_choice == 1:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_dataset_path + '/train/images'
        else:
            images_dir = config.CBIS_DDSM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_dataset_path + '/train/masks'
    elif dataset_choice == 2:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/images'
        else:
            images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_dataset_path + '/train/masks'
    elif dataset_choice == 3:
        if feature_dataset_choice == 10:
            images_dir = config.HAM_dataset_path + '/train/images'
        else:
            images_dir = config.HAM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.HAM_dataset_path + '/train/masks'
    elif dataset_choice == 4:
        if feature_dataset_choice == 10:
            images_dir = config.HAM_CLAHE_dataset_path + '/train/images'
        else:
            images_dir = config.HAM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.HAM_dataset_path + '/train/masks'
    elif dataset_choice == 5:
        if feature_dataset_choice == 10:
            images_dir = config.POLYP_dataset_path + '/train/images'
        else:
            images_dir = config.POLYP_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.POLYP_dataset_path + '/train/masks'
    elif dataset_choice == 6:
        if feature_dataset_choice == 10:
            images_dir = config.POLYP_CLAHE_dataset_path + '/train/images'
        else:
            images_dir = config.POLYP_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.POLYP_dataset_path + '/train/masks'

    dataset = CancerDataset(
        images_dir= images_dir,
        masks_dir= masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    return DataLoader(dataset, batch_size=4, shuffle=True, drop_last = True)

def get_images_dir(dataset_choice, feature_dataset_choice):

    # Create your datasets and data loaders
    images_dir = ''
    
    if dataset_choice == 1:
        if feature_dataset_choice == 10:
            train_images_dir = config.CBIS_DDSM_dataset_path + '/train/images'
            test_images_dir = config.CBIS_DDSM_dataset_path + '/test/images'
        else:
            train_images_dir = config.CBIS_DDSM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.CBIS_DDSM_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.CBIS_DDSM_dataset_path + '/test/masks'
    elif dataset_choice == 2:
        if feature_dataset_choice == 10:
            train_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/images'
            test_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/images'

        else:
            train_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/masks'
    elif dataset_choice == 3:
        if feature_dataset_choice == 10:
            train_images_dir = config.HAM_dataset_path + '/train/images'
            test_images_dir = config.HAM_dataset_path + '/test/images'

        else:
            train_images_dir = config.HAM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.HAM_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.HAM_dataset_path + '/test/masks'
    elif dataset_choice == 4:
        if feature_dataset_choice == 10:
            train_images_dir = config.HAM_CLAHE_dataset_path + '/train/images'
            test_images_dir = config.HAM_CLAHE_dataset_path + '/test/images'

        else:
            train_images_dir = config.HAM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.HAM_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.HAM_dataset_path + '/test/masks'
    elif dataset_choice == 5:
        if feature_dataset_choice == 10:
            train_images_dir = config.POLYP_dataset_path + '/train/images'
            test_images_dir = config.POLYP_dataset_path + '/test/images'

        else:
            train_images_dir = config.POLYP_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.POLYP_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.POLYP_dataset_path + '/test/masks'
    elif dataset_choice == 6:
        if feature_dataset_choice == 10:
            train_images_dir = config.POLYP_CLAHE_dataset_path + '/train/images'
            test_images_dir = config.POLYP_CLAHE_dataset_path + '/test/images'

        else:
            train_images_dir = config.POLYP_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.POLYP_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.POLYP_dataset_path + '/test/masks'
    return train_images_dir,test_images_dir,test_masks_dir

def train_model():
    clear_screen() #clears the terminal screen
    
    #get model and dataset from user
    model_choice = prompt_model()
    clear_screen()
    dataset_choice = prompt_dataset()
    clear_screen()
    feature_dataset_choice = prompt_feature_dataset()

    data_loader = create_data_loader(dataset_choice, feature_dataset_choice)
    train_images_dir, test_images_dir, mask_images_dir = get_images_dir(dataset_choice, feature_dataset_choice)

    if dataset_choice == 1:
        dataset = 'CBIS_DDSM'
    elif dataset_choice == 2:
        dataset = 'CBIS_DDSM_CLAHE'
    elif dataset_choice == 3:
        dataset = 'HAM10000'
    elif dataset_choice == 4:
        dataset = 'HAM10000_CLAHE'
    elif dataset_choice == 5:
        dataset = 'POLYP'
    elif dataset_choice == 6:
        dataset = 'POLYP_CLAHE'

    if model_choice == 1:
        train_deeplab(config.saved_models_path + '/Deeplab/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    elif model_choice == 2:
        train_fcn(config.saved_models_path + '/Fcn/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    elif model_choice == 3:
        train_unet(config.saved_models_path + '/Unet/'+dataset+'/Feature_' + str(feature_dataset_choice), dataset, train_images_dir, test_images_dir, mask_images_dir)

    elif model_choice == 4:
        train_hrnet(config.saved_models_path + '/Hrnet/'+dataset+'/Feature_' + str(feature_dataset_choice), dataset, train_images_dir, test_images_dir, mask_images_dir)

    elif model_choice == 5:
        train_fpn(config.saved_models_path + '/Fpn/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    elif model_choice == 6:
        train_linknet(config.saved_models_path + '/Linknet/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    return