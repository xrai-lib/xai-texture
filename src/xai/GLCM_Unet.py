import torch
from mmseg.apis import init_model, inference_model
import mmcv
import matplotlib.pyplot as plt
import cv2
import config
from torchvision import models, transforms
from PIL import Image

from mmseg.models import build_segmentor
from mmengine import Config
import config
import numpy as np
import mahotas as mh
from skimage import io, color
from utils import add_to_GLCM_Unet_results, add_to_GLCM_results
import os

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

def GLCM(feature_maps):

    # Compute GLCM properties for each feature map
    feature_map_glcm_properties = []
    for i in range(feature_maps.size(1)):
        feature_map = feature_maps[0, i].cpu().detach().numpy()
        glcm_properties = compute_glcm_properties(feature_map)
        feature_map_glcm_properties.append(glcm_properties)

    # Visualize GLCM properties for each feature map
    for i, glcm_properties in enumerate(feature_map_glcm_properties):
        print(f"GLCM Properties for Feature Map {i+1}:")
        for j, prop in enumerate(glcm_properties.mean(axis=0)):
            print(f"Feature {j+1}: {prop}")
            
            
# Define a function to compute GLCM properties for a given image
def compute_glcm_properties(image):
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
def compute_and_visualize_glcm_properties_firstapproach(feature_maps, gray_image):
    num_feature_maps = feature_maps.size(1)
    original_glcm_properties = compute_glcm_properties(gray_image)  # Compute GLCM properties for the original image

    # Initialize lists to store absolute differences for each GLCM property
    absolute_differences = {prop: [] for prop in original_glcm_properties}

    for i in range(num_feature_maps):
        feature_map = feature_maps[:, i:i+1, :, :]  # Extract a single feature map
        feature_map_numpy = feature_map.cpu().detach().numpy()[0, 0]  # Convert to numpy array
        
        # Compute GLCM properties for the feature map
        feature_map_glcm_properties = compute_glcm_properties(feature_map_numpy)
        
        # Compute absolute differences for each GLCM property
        for prop in original_glcm_properties:
            absolute_difference = abs(original_glcm_properties[prop] - feature_map_glcm_properties[prop])
            absolute_differences[prop].append(absolute_difference)

        # Print or compare GLCM properties between the original image and the feature map
        # print(f"GLCM Properties for Feature Map {i+1}:")
        # print("Original Image:", original_glcm_properties)
        # print("Feature Map:", feature_map_glcm_properties)
        # print()  # Add space between feature maps

    # Compute average absolute differences for each GLCM property
    average_absolute_differences = {prop: np.mean(absolute_differences[prop]) for prop in original_glcm_properties}

    # Sort GLCM properties based on average absolute differences
    sorted_properties = sorted(average_absolute_differences, key=average_absolute_differences.get)

    # Print feature ranking
    print("GLCM Feature Ranking:")

    # Print feature ranking
    results = dict()
    for i, prop in enumerate(sorted_properties):
        results[prop] = average_absolute_differences[prop]
    
    return results
        

def preprocess_image(image_path):
    # Move the model to GPU if available
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # Load the image from the file path
    input_image = Image.open(image_path).convert("RGB")
    
    # Define the image transformations
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to 512x512 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])  # Normalize the image tensor
    ])
    
    # Apply transformations and move the image to GPU
    input_image = image_transform(input_image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension
    input_image = input_image.to(device)
    return input_image

def get_feature_maps(model, input_image):
    layers_of_interest = [model.backbone.encoder[0], model.backbone.encoder[1], model.backbone.encoder[2], model.backbone.encoder[3], model.backbone.encoder[4], model.backbone.decoder[0], model.backbone.decoder[1], model.backbone.decoder[2], model.backbone.decoder[3]]  # Add more layers as needed
    
    # Initialize an empty dictionary to store the activations
    activations_dict = {layer.__class__.__name__: [] for layer in layers_of_interest}
    
    # Define the hook function
    def hook_fn(module, input, output):
        activations_dict[module.__class__.__name__].append(output)
    
    # Register the hook function to the desired layers
    for layer in layers_of_interest:
        # Register hook
        hook_handle = layer.register_forward_hook(hook_fn)
        
        # Perform forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_image)
    
        # Remove the hook
        hook_handle.remove()
    
    # Now `activations_dict` contains activations for each layer keyed by their names
    # You can access them like this:
    for layer_name, activations_list in activations_dict.items():
        print(layer_name)
        for i, activations in enumerate(activations_list):
            print(f"Activation {i + 1}:", activations.shape)
    
    featuremaps = [activations_dict['Sequential'][0], activations_dict['Sequential'][1], activations_dict['Sequential'][2], activations_dict['Sequential'][3], activations_dict['Sequential'][4],activations_dict['UpConvBlock'][0],activations_dict['UpConvBlock'][1],activations_dict['UpConvBlock'][2],activations_dict['UpConvBlock'][3]]
    
    return featuremaps

def analyze_GLCM_Unet(model_choice):
    #choice = prompt_dataset()

    config_file = config_file = config.saved_models_path + '/UNET/Feature_10/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
    checkpoint_file = config.saved_models_path + '/UNET/Feature_10/iter_16000.pth'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_file, checkpoint_file, device=device)

    # Load the model using the configuration file
    cfg = Config.fromfile(config_file)
    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_path= config.patch_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    image = io.imread(image_path)

    gray_image = color.rgb2gray(image)

    input_image = preprocess_image(image_path)

    featuremaps = get_feature_maps(model,input_image)
    #all_results = []
    #for i in range(len(featuremaps)):
       #print("First Approach GLCM comparison for Layer_"+str(i))
        #all_results.append(compute_and_visualize_glcm_properties_firstapproach(featuremaps[i],gray_image))

        
        #print(all_results[i])
    #os.makedirs(config.results_path + '/GLCM', exist_ok=True)
    #result_path = config.results_path + '/GLCM/Unet_GLCM_Feature_'+str(choice)+'_dataset.csv'
    #add_to_GLCM_Unet_results(result_path,all_results)
    result_path = config.results_path + '/GLCM/Unet_GLCM.csv'
    results = compute_and_visualize_glcm_properties_firstapproach(featuremaps[8],gray_image)
    print(results)
    add_to_GLCM_results(result_path, results)

