import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from skimage import io, color
import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import functional as F
import config
from utils import add_to_GLCM_results


#Input data preprocess and load to the model starts

# Custom transform for resizing images
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resize(img, self.size)

# Load and preprocess the image
def preprocess_image(image_path, input_size=(512, 512)):
    transform = transforms.Compose([
        ResizeTransform(input_size),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


#image process and load to the models
#Activation feature maps for Layer 1 start

def get_feature_maps(model, input_image):
    activations = []

    def hook(model, input, output):
        activations.append(output)

    # Register hook
    hook_handle = model.backbone.layer4.register_forward_hook(hook)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    # Remove the hook
    hook_handle.remove()

    return activations[0]

#Activation feature maps for Layer 1 end 


# Define a function to compute GLCM properties for a given image
def glcm_properties(image):
    # Convert image to integer type in the range [0, 255]
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Compute GLCM
    glcm = mh.features.haralick(image_uint8)

    # Extract desired GLCM properties
    properties = {
        "ASM": glcm.mean(axis=0)[0],
        "contrast": glcm.mean(axis=0)[1],
        "correlation": glcm.mean(axis=0)[2],
        "variance": glcm.mean(axis=0)[3],
        "IDM": glcm.mean(axis=0)[4],
        "sum_average": glcm.mean(axis=0)[5],
        "sum_entropy": glcm.mean(axis=0)[6],
        "entropy": glcm.mean(axis=0)[7],
        "diff_entropy": glcm.mean(axis=0)[8],
        "IMC1": glcm.mean(axis=0)[9],
        "IMC2": glcm.mean(axis=0)[10],
        "MCC": glcm.mean(axis=0)[11],
        "autocorrelation": glcm.mean(axis=0)[12]
    }
    
    return properties

# Compute and visualize GLCM properties for each feature map
# Compute and visualize GLCM properties for each feature map
def compute_glcm_properties(feature_maps, gray_image):

    num_feature_maps = feature_maps.size(1)
    original_glcm_properties = glcm_properties(gray_image)  # Compute GLCM properties for the original image

    # Initialize lists to store absolute differences for each GLCM property
    absolute_differences = {prop: [] for prop in original_glcm_properties}

    for i in range(num_feature_maps):
        feature_map = feature_maps[:, i:i+1, :, :]  # Extract a single feature map
        feature_map_numpy = feature_map.cpu().detach().numpy()[0, 0]  # Convert to numpy array
        
        # Compute GLCM properties for the feature map
        feature_map_glcm_properties = glcm_properties(feature_map_numpy)
        
        # Compute absolute differences for each GLCM property
        for prop in original_glcm_properties:
            absolute_difference = abs(original_glcm_properties[prop] - feature_map_glcm_properties[prop])
            absolute_differences[prop].append(absolute_difference)

        # Print or compare GLCM properties between the original image and the feature map
        #print(f"GLCM Properties for Feature Map {i+1}:")
        #print("Original Image:", original_glcm_properties)
       # print("Feature Map:", feature_map_glcm_properties)
       # print()  # Add space between feature maps

    # Compute average absolute differences for each GLCM property
    average_absolute_differences = {prop: np.mean(absolute_differences[prop]) for prop in original_glcm_properties}

    # Sort GLCM properties based on average absolute differences
    sorted_properties = sorted(average_absolute_differences, key=average_absolute_differences.get)

    # Print feature ranking
    results = dict()
    for i, prop in enumerate(sorted_properties):
        results[prop] = average_absolute_differences[prop]

    return results

    


def analyse_GLCM(model_choice):
    
    image_path= config.patch_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)

    input_image = preprocess_image(image_path)
    

    # Move the input tensor to GPU if available
    if torch.cuda.is_available():
        input_image = input_image.cuda()

    # Load the model and move it to GPU
    if model_choice == 1:
        result_path = config.results_path + '/GLCM/Deeplab_GLCM.csv'
        model = torch.load(config.saved_models_path + '/Deeplab/Feature_10/deeplab_v3_segmentation.pth')
    elif model_choice == 2:
        result_path = config.results_path + '/GLCM/FCN_GLCM.csv'
        model = torch.load(config.saved_models_path + '/FCN/Feature_10/fcn_resnet101_epoch20_segmentation.pth')

    if torch.cuda.is_available():
        model = model.cuda()

    # Apply the model
    model.eval()
    #with torch.no_grad():
        #output = model(input_image)['out']

    # Example: Get feature maps from 'layer4'
    feature_maps  = get_feature_maps(model, input_image)

    # Compute and visualize GLCM properties for each feature map
    result = compute_glcm_properties(feature_maps, gray_image)

    
    os.makedirs(config.results_path + '/GLCM', exist_ok=True)

    add_to_GLCM_results(result_path, result)
    print('Results are stored in ' + result_path)
