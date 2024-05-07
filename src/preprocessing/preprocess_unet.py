import config
import os
from PIL import Image
import os.path as osp


# Function to convert mask image pixel values
def convert_mask(mask_path, output_path):
    mask = Image.open(mask_path)
    mask = mask.point(lambda p: p // 255)  # Convert pixel value from 255 to 1
    mask.save(output_path)

# Function to list mask image names in a text file
def list_mask_images(directory, output_file):
    mask_images = [os.path.splitext(mask_image)[0] for mask_image in os.listdir(directory) if mask_image.endswith('.png')]
    with open(output_file, 'w') as f:
        for i, mask_image in enumerate(mask_images):
            f.write(mask_image)
            if i != len(mask_images) - 1:  # Check if it's not the last mask image
                f.write('\n')  # Write newline character

def preprocess_dataset_unet_from_patches():

    # Paths
    train_masks_dir = osp.join(config.data_path,'CBIS-DDSM-Patches/train/masks')

    test_masks_dir = osp.join(config.data_path,'CBIS-DDSM-Patches/test/masks')
    segmentation_class_dir = osp.join(config.data_path,'SegmentationClass')
    train_txt_file = osp.join(config.data_path,'train.txt')
    val_txt_file = osp.join(config.data_path,'val.txt')

    # Create SegmentationClass folder if not exists
    if not os.path.exists(segmentation_class_dir):
        os.makedirs(segmentation_class_dir)

    # Process train masks
    for mask_image in os.listdir(train_masks_dir):
        if mask_image.endswith('.png'):
            mask_path = os.path.join(train_masks_dir, mask_image)
            output_path = os.path.join(segmentation_class_dir, mask_image)
            convert_mask(mask_path, output_path)

    # List train mask image names in train.txt
    list_mask_images(train_masks_dir, train_txt_file)

    # Process test masks
    for mask_image in os.listdir(test_masks_dir):
        if mask_image.endswith('.png'):
            mask_path = os.path.join(test_masks_dir, mask_image)
            output_path = os.path.join(segmentation_class_dir, mask_image)
            convert_mask(mask_path, output_path)

    # List test mask image names in val.txt
    list_mask_images(test_masks_dir, val_txt_file)
