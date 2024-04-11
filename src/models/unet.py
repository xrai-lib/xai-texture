import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import calculate_iou, add_to_results
import config

def train_unet(dataset, data_loader):

    # unet model setup
    
    # Move the model to GPU if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        print("CUDA is not available. Training on CPU.")
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 20  # Set the number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iou_scores = []
        
        for batch_idx, (images, masks) in enumerate(data_loader):
            pass

    
    os.makedirs(dataset, exist_ok=True)
    torch.save(model, dataset + '/unet_epoch20_segmentation.pth')
def test_unet(path, choice, data_loader):
    print("Testing Deeplab on Feature: " + str(choice))

    model_path = config.saved_models_path + '/FCN/Feature_' + str(choice) + '/fcn_resnet101_epoch20_segmentation.pth'