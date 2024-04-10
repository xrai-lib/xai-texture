import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from utils import clear_screen
from models.fcn import train_fcn
from models.deeplabv3 import train_deeplab
from models.unet import train_unet
import config

def prompt_model():
    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")

    choice = None
    while True:
            try:
                choice = int(input("Select Model (1-3): "))
                if 1 <= choice <= 3:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 3 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_dataset():

    print("1. Feature 1 (L5E5 / E5L5)")
    print("2. Feature 2 (L5S5 / S5L5)")
    print("3. Feature 3 (L5R5 / L5R5)")
    print("4. Feature 4 (E5S5 / S5E5)")
    print("5. Feature 5 (E5R5 / R5E5)")
    print("6. Feature 6 (R5S5 / S5R5)")
    print("7. Feature 7 (S5S5)")
    print("8. Feature 8 (E5E5)")
    print("9. Feature 9 (R5R5)")
    print("10. CBIS-DDSM-Patches")

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
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

def create_data_loader(dataset_choice):
    
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
    if dataset_choice == 10:
        images_dir = config.patch_dataset_path + '/train/images'
    else:
        images_dir = config.TEM_dataset_path + '/train/textures/Feature_' + str(dataset_choice)

    masks_dir = config.patch_dataset_path + '/train/masks'
    dataset = CancerDataset(
        images_dir= images_dir,
        masks_dir= masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    return DataLoader(dataset, batch_size=8, shuffle=True, drop_last = True)

def train_model():
    clear_screen() #clears the terminal screen
    
    #get model and dataset from user
    model_choice = prompt_model()
    clear_screen()
    dataset_choice = prompt_dataset()

    data_loader = create_data_loader(dataset_choice)

    if model_choice == 1:
         train_deeplab(config.saved_models_path + '/Deeplab/Feature_' + str(dataset_choice), data_loader)
    elif model_choice == 2:
         train_fcn(config.saved_models_path + '/FCN/Feature_' + str(dataset_choice), data_loader)
    else:
         train_unet(config.saved_models_path + '/UNET/Feature_' + str(dataset_choice), data_loader)
         
    return