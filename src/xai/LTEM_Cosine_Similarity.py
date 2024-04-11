import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import config

# Define a function to load the models
def load_model(model_path):

    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model on CPU
    model.eval()
    return model

# Define a function to get feature maps using hook
def get_feature_maps(model, input_image, layer_name):
    activations = []

    def hook(model, input, output):
        activations.append(output)

    # Register hook
    target_layer = getattr(model.backbone, layer_name)
    hook_handle = target_layer.register_forward_hook(hook)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    # Remove the hook
    hook_handle.remove()

    return activations[0]

def cosine_similarity(model):

    # Prepare a test image
    test_image_path = config.patch_dataset_path + 'test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
    test_image = Image.open(test_image_path).convert("RGB")

    # Define a transform for the test image
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transform to the test image
    test_image = test_transform(test_image).unsqueeze(0)

    # Load both trained models

    if model == 0:
        model1 = load_model('deeplabv3_complete_model.pth')
        model2 = load_model('feature training from MLOPS/deeplabv3_model_texture2_out_of_9.pth')

    # Compare the feature maps from different layers
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in layers:
        # Get feature maps from the specified layer of both models
        with torch.no_grad():
            features1 = get_feature_maps(model1, test_image, layer_name).cpu().squeeze(0)
            features2 = get_feature_maps(model2, test_image, layer_name).cpu().squeeze(0)

        # Calculate the similarities using cosine similarity
        similarities = []
        for i in range(features1.size(0)):  # Iterate over each feature map
            similarity = cosine_similarity(features1[i].reshape(1, -1), features2[i].reshape(1, -1))[0, 0]
            similarities.append(similarity)

        # Calculate the average similarity
        average_similarity = np.mean(similarities)
        print(f"Average cosine similarity between feature maps in {layer_name}: {average_similarity:.4f}")