import torch
from mmseg.apis import init_model, inference_model
import mmcv
import numpy as np
import config
import os
import csv

from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import config
from utils import add_to_LTEM_unet_results
from pathlib import Path
import matplotlib.pyplot as plt

def preprocess_sampleimage(choice_dataset):
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if choice_dataset == 1:
        image_path= config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        texture_image_paths = [config.CBIS_DDSM_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
        dataset = 'CBIS_DDSM'
    elif choice_dataset == 2:
        image_path= config.CBIS_DDSM_CLAHE_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        texture_image_paths = [config.CBIS_DDSM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
        dataset = 'CBIS_DDSM_CLAHE'
    elif choice_dataset == 3:
        image_path= config.HAM_dataset_path + '/test/images/ISIC_0024306.jpg'
        texture_image_paths = [config.HAM_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]
        dataset = 'HAM10000'
    elif choice_dataset == 4:
        image_path= config.HAM_CLAHE_dataset_path + '/test/images/ISIC_0024306.jpg'
        texture_image_paths = [config.HAM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]
        dataset = 'HAM10000_CLAHE'
    elif choice_dataset == 5:
        image_path= config.POLYP_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'
        texture_image_paths = [config.HAM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]
        dataset = 'POLYP'
    elif choice_dataset == 6:
        image_path= config.POLYP_CLAHE_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'
        texture_image_paths = [config.HAM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]
        dataset = 'POLYP_CLAHE'

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

    return input_image, image_path, dataset, texture_image_paths

def get_feature_maps_original_model(input_image, choice_dataset):
    # Original Model
    # Load the pre-trained model
    input_image, image_path, dataset, texture_image_paths = preprocess_sampleimage(choice_dataset)
    config_file_original = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_10/fcn_hr18_4xb2-160k_cityscapes-512x1024.py'
    checkpoint_file_original = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_10/iter_16000.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_model = init_model(config_file_original, checkpoint_file_original, device=device)

    checkpoint = torch.load(checkpoint_file_original)
    original_model.load_state_dict(checkpoint['state_dict'])

    # Set the model to evaluation mode
    original_model.eval()

    layers_of_interest_original_model = [original_model.backbone.conv1, original_model.backbone.bn1, original_model.backbone.conv2, original_model.backbone.bn2, original_model.backbone.relu, original_model.backbone.layer1[0], original_model.backbone.layer1[1], original_model.backbone.layer1[2], original_model.backbone.layer1[3], original_model.backbone.transition1[0], original_model.backbone.transition1[1], original_model.backbone.stage2[0], original_model.backbone.transition2[2], original_model.backbone.stage3[0], original_model.backbone.stage3[1], original_model.backbone.stage3[2], original_model.backbone.stage3[3], original_model.backbone.transition3[3], original_model.backbone.stage4[0], original_model.backbone.stage4[1], original_model.backbone.stage4[2]]  # Add more layers as needed

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
    featuremaps_original = [activations_dict_original['Conv2d'][0], activations_dict_original['Conv2d'][1], activations_dict_original['SyncBatchNorm'][0], 
        activations_dict_original['SyncBatchNorm'][1], activations_dict_original['ReLU'][0], activations_dict_original['ReLU'][1], 
        activations_dict_original['Bottleneck'][0], activations_dict_original['Bottleneck'][1], activations_dict_original['Bottleneck'][2], 
        activations_dict_original['Bottleneck'][3], activations_dict_original['Sequential'][0], activations_dict_original['Sequential'][1], 
        activations_dict_original['Sequential'][2], activations_dict_original['Sequential'][3], activations_dict_original['HRModule'][0][0], 
        activations_dict_original['HRModule'][0][1], activations_dict_original['HRModule'][1][0], activations_dict_original['HRModule'][1][1], 
        activations_dict_original['HRModule'][1][2], activations_dict_original['HRModule'][2][0], activations_dict_original['HRModule'][2][1], 
        activations_dict_original['HRModule'][2][2], activations_dict_original['HRModule'][3][0], activations_dict_original['HRModule'][3][1],
        activations_dict_original['HRModule'][3][2], activations_dict_original['HRModule'][4][0], activations_dict_original['HRModule'][4][1], 
        activations_dict_original['HRModule'][4][2], activations_dict_original['HRModule'][5][0], activations_dict_original['HRModule'][5][1], 
        activations_dict_original['HRModule'][5][2], activations_dict_original['HRModule'][5][3], activations_dict_original['HRModule'][6][0], 
        activations_dict_original['HRModule'][6][1], activations_dict_original['HRModule'][6][2], activations_dict_original['HRModule'][6][3], 
        activations_dict_original['HRModule'][7][0], activations_dict_original['HRModule'][7][1], activations_dict_original['HRModule'][7][2], 
        activations_dict_original['HRModule'][7][3]]
    return featuremaps_original


# Feature Model
# Load the pre-trained model


# Extract feature maps for an image
def extract_feature_maps(image_path, model, target_layers):
    # Load and preprocess the image

    # Hook function to extract feature maps
    feature_maps = {}
    def hook_fn(module, input, output, name):
        feature_maps[name] = output

    # Register hooks to specific layers of the model
    def register_hooks(model, target_layers):
        hooks = []
        for name, layer in model.named_modules():
            if name in target_layers:
                hooks.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name)))
                #print(f"Hook registered for layer: {name}")
        return hooks

    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to 512x512 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])  # Normalize the image tensor
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_image = image_transform(image).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        input_image = input_image.cuda()

    # Initialize feature maps dictionary
    feature_maps = {}

    # Register hooks
    hooks = register_hooks(model, target_layers)

    # Forward pass through the model
    with torch.no_grad():
        _ = model(input_image)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return feature_maps

# Define a function to compute the cosine similarity
def compute_cosine_similarity(feature_map, texture_feature_map):
    similarities = []
    for i in range(feature_map.size(1)):
        fm = feature_map[0, i].cpu().detach().numpy().flatten()
        tex_fm = texture_feature_map[0, i].cpu().detach().numpy().flatten()
        similarity = cosine_similarity(fm.reshape(1, -1), tex_fm.reshape(1, -1))
        similarities.append(similarity[0][0])
    return similarities


def LTEM_analysis_hrnet_textureimages(choice_dataset, dataset):

    input_image, test_image_path, dataset, texture_image_paths = preprocess_sampleimage(choice_dataset)
        # Define the image transformations
    
    # Specify the path to your input image
    #test_image_path = '../Unet/dataset_cancer/dataset_traintestfolder_mask255_MassTrainingonly/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    # Load the pre-trained model
    config_file_original = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_10/fcn_hr18_4xb2-160k_cityscapes-512x1024.py'  # Specify the path to your config file
    checkpoint_file_original = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_10/iter_16000.pth'  # Specify the path to your checkpoint file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_model = init_model(config_file_original, checkpoint_file_original, device=device)
    checkpoint = torch.load(checkpoint_file_original)
    original_model.load_state_dict(checkpoint['state_dict'])
    original_model.eval()  # Set the model to evaluation mode

    # List the layers of interest
    target_layers = [
        'backbone.conv1', 
        'backbone.bn1', 
        'backbone.conv2', 
        'backbone.bn2', 
        'backbone.relu', 
        'backbone.layer1.0', 
        'backbone.layer1.1', 
        'backbone.layer1.2', 
        'backbone.layer1.3', 
        'backbone.transition1.0', 
        'backbone.transition1.1', 
        'backbone.stage2.0', 
        'backbone.transition2.2', 
        'backbone.stage3.0', 
        'backbone.stage3.1', 
        'backbone.stage3.2', 
        'backbone.stage3.3', 
        'backbone.transition3.3', 
        'backbone.stage4.0', 
        'backbone.stage4.1', 
        'backbone.stage4.2'
    ]  # Add more layers as needed

    # Extract feature maps for the test image
    test_feature_maps = extract_feature_maps(test_image_path, original_model, target_layers)
    #test_feature_maps = get_feature_maps_original_model(input_image, choice_dataset)

    # Initialize a dictionary to store the average similarities
    layer_similarities = {layer: [] for layer in target_layers}

    # Load and preprocess texture images
    #texture_image_paths = [f'../Unet/dataset_cancer/dataset_traintestfolder_mask255_MassTrainingonly/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]

    # Extract feature maps and compute cosine similarities for each texture image
    for i, texture_path in enumerate(texture_image_paths):
        texture_feature_maps = extract_feature_maps(texture_path, original_model, target_layers)
        #print("texture_feature_maps", texture_feature_maps)

        print(f"\nProcessing texture image {i + 1}")

        for layer_name, test_feature_map in test_feature_maps.items():
            texture_feature_map = texture_feature_maps[layer_name]
            if len(texture_feature_map) > 1:
                avg_similarities = []
                for j in range(len(texture_feature_map)):
                    similarities = compute_cosine_similarity(test_feature_map[j], texture_feature_map[j])   
                    avg_similarity = np.mean(similarities)
                    avg_similarities.append(avg_similarity)
                
                layer_similarities[layer_name].append(np.mean(avg_similarities))
                print(f"Layer {layer_name} - Texture {i + 1} Average Cosine Similarity: {np.mean(avg_similarities)}")        
            else:
                similarities = compute_cosine_similarity(test_feature_map, texture_feature_map)       
                avg_similarity = np.mean(similarities)
                layer_similarities[layer_name].append(avg_similarity)
                print(f"Layer {layer_name} - Texture {i + 1} Average Cosine Similarity: {avg_similarity}")

    
    result_path = config.results_path + '/LTEM/Hrnet/' + dataset + '/' + dataset + '_Hrnet_LTEM.png'

    # Plot the results
    plt.figure(figsize=(15, 8))

    for layer_name, similarities in layer_similarities.items():
        plt.plot(range(1, len(texture_image_paths) + 1), similarities, label=layer_name)

    plt.xlabel('Texture Image')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Average Cosine Similarity by Layer and Texture Image')
    plt.legend()
    plt.xticks(range(1, len(texture_image_paths) + 1))
    plt.savefig(Path(result_path), format='png', dpi=300, bbox_inches='tight')
    plt.show()

def LTEM_analysis_hrnet(choice_dataset):
    result_list = []
    input_image, test_image_path, dataset, texture_image_paths = preprocess_sampleimage(choice_dataset)
    for j in range(9):
        featuremaps_original = get_feature_maps_original_model(input_image, choice_dataset)
        
        config_file_feature = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_' + str(j+1)+'/fcn_hr18_4xb2-160k_cityscapes-512x1024.py'
        checkpoint_file_feature = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_' + str(j+1)+'/iter_16000.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_model = init_model(config_file_feature, checkpoint_file_feature, device=device)

        checkpoint_feature = torch.load(checkpoint_file_feature)
        feature_model.load_state_dict(checkpoint_feature['state_dict'])

        # Set the model to evaluation mode
        feature_model.eval()

        layers_of_interest_feature_model = [feature_model.backbone.conv1, feature_model.backbone.bn1, feature_model.backbone.conv2, feature_model.backbone.bn2, feature_model.backbone.relu, feature_model.backbone.layer1[0], feature_model.backbone.layer1[1], feature_model.backbone.layer1[2], feature_model.backbone.layer1[3], feature_model.backbone.transition1[0], feature_model.backbone.transition1[1], feature_model.backbone.stage2[0], feature_model.backbone.transition2[2], feature_model.backbone.stage3[0], feature_model.backbone.stage3[1], feature_model.backbone.stage3[2], feature_model.backbone.stage3[3], feature_model.backbone.transition3[3], feature_model.backbone.stage4[0], feature_model.backbone.stage4[1], feature_model.backbone.stage4[2]]  # Add more layers as needed

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



        featuremaps_feature = [activations_dict_feature['Conv2d'][0], activations_dict_feature['Conv2d'][1], activations_dict_feature['SyncBatchNorm'][0], 
        activations_dict_feature['SyncBatchNorm'][1], activations_dict_feature['ReLU'][0], activations_dict_feature['ReLU'][1], 
        activations_dict_feature['Bottleneck'][0], activations_dict_feature['Bottleneck'][1], activations_dict_feature['Bottleneck'][2], 
        activations_dict_feature['Bottleneck'][3], activations_dict_feature['Sequential'][0], activations_dict_feature['Sequential'][1], 
        activations_dict_feature['Sequential'][2], activations_dict_feature['Sequential'][3], activations_dict_feature['HRModule'][0][0], 
        activations_dict_feature['HRModule'][0][1], activations_dict_feature['HRModule'][1][0], activations_dict_feature['HRModule'][1][1], 
        activations_dict_feature['HRModule'][1][2], activations_dict_feature['HRModule'][2][0], activations_dict_feature['HRModule'][2][1], 
        activations_dict_feature['HRModule'][2][2], activations_dict_feature['HRModule'][3][0], activations_dict_feature['HRModule'][3][1],
        activations_dict_feature['HRModule'][3][2], activations_dict_feature['HRModule'][4][0], activations_dict_feature['HRModule'][4][1], 
        activations_dict_feature['HRModule'][4][2], activations_dict_feature['HRModule'][5][0], activations_dict_feature['HRModule'][5][1], 
        activations_dict_feature['HRModule'][5][2], activations_dict_feature['HRModule'][5][3], activations_dict_feature['HRModule'][6][0], 
        activations_dict_feature['HRModule'][6][1], activations_dict_feature['HRModule'][6][2], activations_dict_feature['HRModule'][6][3], 
        activations_dict_feature['HRModule'][7][0], activations_dict_feature['HRModule'][7][1], activations_dict_feature['HRModule'][7][2], 
        activations_dict_feature['HRModule'][7][3]]


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
    result_path = config.results_path + '/LTEM/Hrnet/' + dataset + '/' + dataset + '_Hrnet_LTEM.csv'
    os.makedirs(Path(result_path).parent, exist_ok=True)
    add_to_LTEM_unet_results(result_path,result_list)
    LTEM_analysis_hrnet_textureimages(choice_dataset, dataset)


