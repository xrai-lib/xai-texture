# GLCM Analysis
This Python script facilitates the analysis of texture features using Gray Level Co-occurrence Matrix (GLCM) properties. GLCM features provide a way to characterize the texture of an image by considering the spatial relationship of pixels. The script extracts GLCM properties for both the original image and feature maps obtained from trained models, allowing for a comparison of texture characteristics. For this we used Mahotas.

Luis Pedro Coelho. "Mahotas: Open source software for scriptable computer vision" in Journal of Open Research Software, vol 1, 2013.
- **DOI**: [10.5334/jors.ac](https://doi.org/10.5334/jors.ac)


## Description

The script provides the following functionality:

1. **preprocess_image**: This function preprocesses the input image by resizing it and applying normalization transformations required for model input.

2. **get_feature_maps**: This function extracts feature maps from a specified layer of the model. In this script, it extracts feature maps from the 'layer4' of the model.

3. **glcm_properties**: This function computes GLCM properties for a given image. It utilizes the mahotas library to calculate GLCM properties such as ASM, contrast, correlation, etc.

4. **compute_glcm_properties**: This function computes GLCM properties for each feature map obtained from the model. It then calculates the absolute differences between the GLCM properties of the original image and each feature map to identify the most important GLCM features.

5. **analyse_GLCM**: This function orchestrates the GLCM analysis process. It loads the input image, extracts feature maps using a specified model, computes GLCM properties, and saves the results to a CSV file.

## GLCM Properties

The GLCM properties extracted by the script include:

1. Angular Second Moment (ASM) or Energy
2. Contrast
3. Correlation
4. Variance
5. Inverse Difference Moment (IDM) or Homogeneity
6. Sum Average
7. Sum Entropy
8. Entropy
9. Difference Entropy
10. Information Measure of Correlation 1 (IMC1)
11. Information Measure of Correlation 2 (IMC2)
12. Maximum Correlation Coefficient (MCC)
13. Autocorrelation

These properties capture different aspects of the image's texture and can be used for various image processing and analysis tasks.

## Usage

To use this script, specify the model choice (1 for Deeplab, 2 for FCN, 3 for UNet), and call the `analyse_GLCM` function for (Deeplabv3 and FCN), and `analyze_GLCM_Unet` function for Unet. The script will compute GLCM properties for the original image and feature maps obtained from the chosen model and save the results to a CSV file.

Additionally, the `config` module and `utils.py` script are imported to access paths, configurations, and utility functions relevant to the project.

## References

- [GLCM-Based Feature Extraction and Medical X-Ray Image Classification](https://www.springerprofessional.de/en/glcm-based-feature-extraction-and-medical-x-ray-image-classifica/25596534)


# Cosine Similarity Analysis for Laws' Texture Energy Measures (LTEM)

This Python script conducts a cosine similarity analysis to compare feature maps obtained from a model trained on Laws' Texture Energy Measures (LTEM) datasets with those from the original model trained on the original dataset. The cosine similarity metric measures the cosine of the angle between two vectors and is often used to assess the similarity between feature representations.

## Description

The script provides the following functionality:

1. **load_model**: This function loads a trained model from the specified path.

2. **get_feature_maps**: This function extracts feature maps from a specified layer of the model using a hook mechanism.

3. **calculate_cosine_similarity**: This function calculates the cosine similarity between feature maps obtained from two different models. It computes the average cosine similarity across multiple layers of the models.

4. **cosine_similarity_analysis**: This function orchestrates the cosine similarity analysis process. It iterates over different LTEM models (trained on 9 different datasets) and compares their feature maps with those of the original model.

## LTEM Overview

Laws' Texture Energy Measures (LTEM) is a robust method for extracting secondary features from images. It employs a series of 5x5 convolution masks (L5, E5, S5, R5) to compute texture energy, generating a vector of nine values for each pixel analyzed. These values correspond to four key characteristics of the image: level, edge, spot, and ripple. The analysis culminates in nine feature vectors representing different texture characteristics.

## Cosine Similarity Analysis

The cosine similarity analysis involves the following steps:

1. **Preprocessing**: A test image is prepared and transformed using normalization and resizing operations.

2. **Model Loading**: Both the original model (trained on the original dataset) and the LTEM model (trained on a specific LTEM dataset) are loaded.

3. **Feature Extraction**: Feature maps are extracted from specified layers of both models.

4. **Cosine Similarity Calculation**: Cosine similarity is calculated between corresponding feature maps from both models across multiple layers.

5. **Result Storage**: The average cosine similarity for each layer is computed and stored in a CSV file for further analysis.

## Usage

To use this script, specify the model choice (1 for Deeplab, 2 for FCN, 3 for UNet) and call the `cosine_similarity_analysis` function for (Deeplabv3 and FCN) and `LTEM_analysis_unet` function for UNet model. The script will conduct cosine similarity analysis for LTEM models trained on different datasets compared to the original model.


Additionally, the `config` module and `utils.py` script are imported to access paths, configurations, and utility functions relevant to the project.

## References

- [Laws' Texture Energy Measures (LTEM)](https://www.sciencedirect.com/science/article/pii/S1877050915018700)



