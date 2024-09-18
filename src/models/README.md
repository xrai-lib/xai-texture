# DeepLabv3 Segmentation Training and Testing

## Overview
This repository contains Python code for training and testing the DeepLabv3 model for semantic image segmentation. Semantic segmentation involves classifying each pixel in an image into a specific category, which is particularly useful in tasks such as object detection, scene understanding, and medical image analysis.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- pandas

## Configuration
### Training
The `train_deeplab()` function sets up and trains the DeepLabv3 model using a specified dataset and data loader. It initializes the model with pre-trained weights and fine-tunes it for semantic segmentation tasks. The model architecture is based on a ResNet-101 backbone with a modified classifier head for single-class segmentation. The training process utilizes the Adam optimizer with a learning rate of 1e-4 and binary cross-entropy loss. The training loop runs for 20 epochs, during which it prints the loss and Intersection over Union (IoU) metric for each batch.

### Testing
The `test_deeplab()` function evaluates the trained DeepLabv3 model on a test dataset using a specified data loader. It loads the trained model from the saved checkpoint file and computes the IoU metric for each image in the test set. The average IoU score is then calculated and added to the test results.

## Usage
1. **Training:** To train the DeepLabv3 model, call the `train_deeplab()` function with the appropriate dataset and data loader.
   
2. **Testing:** To test the trained model, call the `test_deeplab()` function with the path to the saved model checkpoint, the feature choice (if applicable), and the test data loader.


# FCN (Fully Convolutional Network) Training and Testing

This repository contains Python code for training and testing a Fully Convolutional Network (FCN) for semantic segmentation tasks using PyTorch. FCN is a popular architecture for image segmentation tasks, capable of producing pixel-level segmentation masks for various classes in an image.

## Prerequisites
- Python 3.x
- PyTorch
- torchvision
- NumPy

## Configuration
- **Training Configuration**: The `train_fcn` function trains the FCN model using a specified dataset and data loader. It utilizes a pre-trained FCN-ResNet101 model for feature extraction and modifies the classifier head for single-class segmentation. Training parameters such as the number of epochs, optimizer (Adam), and loss function (Binary Cross-Entropy with Logits) are configurable.

- **Testing Configuration**: The `test_fcn` function evaluates the trained FCN model on a test dataset using a specified data loader. It loads the trained model, moves it to the GPU if available, and computes the Intersection over Union (IoU) score to evaluate segmentation performance. Test results are saved in a designated directory for further analysis.

## Description
- **Training**: During training, the FCN model undergoes multiple epochs of forward and backward passes. The model parameters are optimized using the Adam optimizer, and the loss is computed based on the predicted segmentation masks and ground truth masks. The average loss and IoU score are calculated for each epoch to monitor training progress.

- **Testing**: After training, the trained FCN model is tested on a separate test dataset. The model is evaluated on its ability to accurately segment objects in the images. The average IoU score is calculated to assess the segmentation performance, and the results are saved for analysis.

# UNet Training and Testing

This repository contains Python code for training and testing a UNet model for semantic segmentation tasks using PyTorch. UNet is a popular architecture for image segmentation, known for its symmetric encoder-decoder structure and skip connections that help preserve spatial information.

## Prerequisites
- Python 3.x
- PyTorch
- NumPy

## Configuration
- **Training Configuration**: The `train_unet` function trains the UNet model using a specified dataset and data loader. The model architecture consists of an encoder-decoder network with contracting and expanding blocks. Training parameters such as the number of epochs, optimizer (Adam), and loss function (Binary Cross-Entropy with Logits) are configurable.

- **Testing Configuration**: The `test_unet` function evaluates the trained UNet model on a test dataset using a specified data loader. It loads the trained model and computes performance metrics such as Intersection over Union (IoU) to evaluate segmentation accuracy. Test results are saved for analysis.

## Description
- **Model Architecture**: The UNet model consists of an encoder section that gradually reduces spatial dimensions and increases feature channels, followed by a decoder section that upsamples the features back to the original input resolution. Skip connections between corresponding encoder and decoder layers help retain spatial information.

- **Training Process**: During training, the UNet model undergoes multiple epochs of forward and backward passes. The model parameters are optimized using the Adam optimizer, and the loss is computed based on the predicted segmentation masks and ground truth masks. The average loss and IoU score are calculated for each epoch to monitor training progress.

- **Testing Process**: After training, the trained UNet model is tested on a separate test dataset. The model's ability to accurately segment objects in the images is evaluated using performance metrics such as IoU. Test results are saved for further analysis.

## Acknowledgments
- This code is adapted from various open-source repositories and tutorials on semantic segmentation using PyTorch.
- Special thanks to the PyTorch community for their contributions and support.



