
import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import calculate_iou, add_to_results
import config

def train_fcn(dataset, data_loader):

    # FCN model setup
    model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))  # Change the classifier head for 1 class


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
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)['out']

            # Squeeze the model's output to remove the channel dimension (if present)
            outputs = outputs.squeeze(1)

            # Ensure that the target masks have a single channel
            if masks.dim() == 4:
                masks = masks[:, 0, :, :]  # Take the first channel

            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iou_score = calculate_iou(torch.sigmoid(outputs), masks)
            iou_scores.append(iou_score)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], "
                    f"Loss: {loss.item():.4f}, IoU: {iou_score:.4f}")

        epoch_loss = running_loss / len(data_loader)
        epoch_iou = np.mean(iou_scores)
        print(f"End of Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Average IoU: {epoch_iou:.4f}")

    print('Finished Training')
    
    os.makedirs(dataset, exist_ok=True)
    torch.save(model, dataset + '/fcn_resnet101_epoch20_segmentation.pth')

def test_fcn(result_path, choice, data_loader):
    print("Testing FCN on Feature: " + str(choice))

    # Load the downloaded model
    model = torch.load(config.saved_models_path + '/FCN/Feature_' + str(choice) + '/fcn_resnet101_epoch20_segmentation.pth', map_location=torch.device('cpu'))

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Evaluate the model on the test dataset
    model.eval()
    iou_scores = []

    for images, masks in data_loader:
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)['out']

        iou_score = calculate_iou(torch.sigmoid(outputs), masks)
        iou_scores.append(iou_score)

    average_iou = np.mean(iou_scores)

    os.makedirs(config.results_path, exist_ok=True)
    add_to_results(result_path, choice, average_iou)

    print(f"Testing Successful. Results added to " + result_path)