# CBIS-DDSM-Patches Dataset Downloader

This Python script facilitates the download and extraction of the CBIS-DDSM-Patches dataset. The CBIS-DDSM-Patches dataset contains image patches extracted from the larger CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset, which is commonly used for breast cancer detection and diagnosis tasks.

## Description

The script provides the following functionality:

1. **download_file**: This function takes a URL pointing to a file and a local path where the file should be saved. It downloads the file from the specified URL in chunks and saves it locally.

2. **generate_patch_dataset**: This function orchestrates the download and extraction process for the CBIS-DDSM-Patches dataset. It first specifies the URL of the ZIP file containing the dataset and the local path where it should be saved. Then, it downloads the ZIP file using the `download_file` function. Finally, it extracts all the contents of the ZIP file into the specified destination directory, which typically resides within the project's data directory.

## Usage

To use this script, simply call the `generate_patch_dataset` function. It will automatically download the CBIS-DDSM-Patches dataset ZIP file from the specified URL and extract its contents into the designated directory.

## Dependencies

The script relies on the `requests` library for sending HTTP requests and the `zipfile` module for ZIP file extraction. Additionally, it imports the `config` module, which contains paths and configurations relevant to the project.

# Texture Extraction using LAWS Filters

This Python script facilitates the extraction of texture features from images using Laws' Texture Energy Measures (LAWS) filters. LAWS filters are a set of 5x5 convolution masks designed to compute texture energy, allowing for comprehensive texture analysis within image datasets.

## Description

The script provides the following functionality:

1. **apply_convolution**: This function applies convolution with a given kernel to an input image using OpenCV's `filter2D` method.

2. **generate_TEM_dataset**: This function generates a dataset of texture features using LAWS filters. It iterates over images in the input folder, applies LAWS filters to calculate texture features, and saves the resulting features as images in the output directory.

## Texture Features

In the realm of image analysis, texture features are pivotal for classification tasks. LAWS filters generate texture features based on four key characteristics: level, edge, spot, and ripple. Each LAWS filter corresponds to a specific combination of these characteristics. The chosen combinations for texture features are as follows:

1. L5E5 / E5L5
2. R5S5 / S5R5
3. L5S5 / S5L5
4. L5R5 / R5L5
5. E5S5 / S5E5
6. E5R5 / R5E5
7. S5S5
8. R5R5
9. E5E5

## Usage

To use this script, call the `generate_TEM_dataset` function, specifying whether the dataset to be generated is for testing or training. The script will automatically calculate texture features using LAWS filters for images in the input directory and save the resulting features as images in the output directory.

## Dependencies

The script requires the OpenCV library (`cv2`) for image processing and manipulation. Additionally, it relies on the `numpy` library for numerical operations. The `config` module is imported to access paths and configurations relevant to the project.





