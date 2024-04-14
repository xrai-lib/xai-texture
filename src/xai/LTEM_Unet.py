import torch
from mmseg.apis import init_model, inference_model
import mmcv
import numpy as np
import os
import csv

from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import config
from utils import add_to_LTEM_unet_results


def get_image():
     # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_path= config.patch_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

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

def get_feature_maps_original_model():
    
    input_image = get_image()
    # Original Model
    # Load the pre-trained model
    config_file_original = config.saved_models_path + '/UNET/Feature_10/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
    checkpoint_file_original = config.saved_models_path + '/UNET/Feature_10/iter_16000.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_model = init_model(config_file_original, checkpoint_file_original, device=device)

    checkpoint = torch.load(checkpoint_file_original)
    original_model.load_state_dict(checkpoint['state_dict'])

    # Set the model to evaluation mode
    original_model.eval()

    layers_of_interest_original_model = [original_model.backbone.encoder[0], original_model.backbone.encoder[1], original_model.backbone.encoder[2], original_model.backbone.encoder[3], original_model.backbone.encoder[4], original_model.backbone.decoder[0], original_model.backbone.decoder[1], original_model.backbone.decoder[2], original_model.backbone.decoder[3]]  # Add more layers as needed

    # Initialize an empty dictionary to store the activations
    activations_dict_original= {layer.__class__.__name__: [] for layer in layers_of_interest_original_model}

    # Define the hook function
    def hook_fn(module, input, output):
        activations_dict_original[module.__class__.__name__].append(output)

    # Register the hook function to the desired layers
    for layer in layers_of_interest_original_model:
        # Register hook
        hook_handle = layer.register_forward_hook(hook_fn)
        
        # Perform forward pass
        original_model.eval()
        with torch.no_grad():
            _ = original_model(input_image)

        # Remove the hook
        hook_handle.remove()

    

    #print("Original Model", activations_dict_original['Sequential'][0])
    featuremaps_original = [activations_dict_original['Sequential'][0], activations_dict_original['Sequential'][1], activations_dict_original['Sequential'][2], activations_dict_original['Sequential'][3], activations_dict_original['Sequential'][4],activations_dict_original['UpConvBlock'][0],activations_dict_original['UpConvBlock'][1],activations_dict_original['UpConvBlock'][2],activations_dict_original['UpConvBlock'][3]]
    return featuremaps_original


# Feature Model
# Load the pre-trained model
result_list = []

def LTEM_analysis_unet():
    input_image = get_image()
    for j in range(9):
        featuremaps_original = get_feature_maps_original_model()
        
        config_file_feature = config.saved_models_path + '/UNET/Feature_' + str(j+1)+'/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
        checkpoint_file_feature = config.saved_models_path + '/UNET/Feature_' + str(j+1)+'/iter_16000.pth'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature_model = init_model(config_file_feature, checkpoint_file_feature, device=device)

        checkpoint_feature = torch.load(checkpoint_file_feature)
        feature_model.load_state_dict(checkpoint_feature['state_dict'])

        # Set the model to evaluation mode
        feature_model.eval()

        layers_of_interest_feature_model = [feature_model.backbone.encoder[0], feature_model.backbone.encoder[1], feature_model.backbone.encoder[2], feature_model.backbone.encoder[3], feature_model.backbone.encoder[4], feature_model.backbone.decoder[0], feature_model.backbone.decoder[1], feature_model.backbone.decoder[2], feature_model.backbone.decoder[3]]  # Add more layers as needed

        # Initialize an empty dictionary to store the activations
        activations_dict_feature= {layer.__class__.__name__: [] for layer in layers_of_interest_feature_model}

        # Define the hook function
        def hook_fn(module, input, output):
            activations_dict_feature[module.__class__.__name__].append(output)

        # Register the hook function to the desired layers
        for layer in layers_of_interest_feature_model:
            # Register hook
            hook_handle = layer.register_forward_hook(hook_fn)
            
            # Perform forward pass
            feature_model.eval()
            with torch.no_grad():
                _ = feature_model(input_image)

            # Remove the hook
            hook_handle.remove()



        featuremaps_feature = [activations_dict_feature['Sequential'][0], activations_dict_feature['Sequential'][1], activations_dict_feature['Sequential'][2], activations_dict_feature['Sequential'][3], activations_dict_feature['Sequential'][4],activations_dict_feature['UpConvBlock'][0],activations_dict_feature['UpConvBlock'][1],activations_dict_feature['UpConvBlock'][2],activations_dict_feature['UpConvBlock'][3]]


        # for encoder layers

        #for i in range(len(activations_dict_feature['Sequential'])):
        #   for j in range(len(activations_dict_feature['Sequential'][i][0]))

        
        results = []
        for i in range(len(featuremaps_original)):
            num_feature_maps = featuremaps_original[i].size(1)
            #print(num_feature_maps)
            similarities = []
            for j in range(num_feature_maps):
                similarity = cosine_similarity(featuremaps_original[i][0,j].cpu().detach().numpy().reshape(1, -1), featuremaps_feature[i][0,j].cpu().detach().numpy().reshape(1, -1))[0, 0]
                similarities.append(similarity)
            #print(featuremaps_original[i][0, num_feature_maps-1])

            # Calculate the average similarity
            average_similarity = np.mean(similarities)
            print(f"Average cosine similarity between feature maps in Activation {i}: {average_similarity:.4f}")
            results.append(average_similarity)
        result_list.append(results)

    result_path = config.results_path + '/LTEM_Cosine_Similarity/Unet_LTEM.csv'
    add_to_LTEM_unet_results(result_path,result_list)



