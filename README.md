# Explainability in Deep Learning Segmentation Models for Breast Cancer by Analogy with Texture Analysis

This repository hosts the source code and related materials for the paper titled "Explainability in Deep Learning Segmentation Models for Breast Cancer by Analogy with Texture Analysis". Our project aims to advance the explainability of deep learning models in medical imaging, specifically in the context of breast cancer segmentation. By drawing analogies with texture analysis, we propose a novel approach to interpret the model's decisions, making these models more transparent and trustworthy for medical practitioners.

## Prerequisites

Before setting up the project, ensure you have the following installed on your system:
- Git
- Anaconda

## Required Libraries and Installation steps

### Create conda environment with python>3.7
```bash
conda create --name xai python=3.8
conda activate xai
```

### Install pytorch [official instructions](https://pytorch.org/get-started/locally/) according to your CUDA support.

#### For Linux and Windows

The version used for the experiments for the paper is given below:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
``` 
#### For Mac

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```
### Install mmcv with mim  
```bash
pip install -U openmim
mim install mmengine==0.10.3
mim install mmcv==2.1.0
```
### Install mmsegmentation with pip  
```bash
pip install mmsegmentation==1.2.2  
```
### Install other dependencies 
```bash
pip install ftfy regex mahotas scikit-image scikit-learn opencv-python pathlib segmentation-models-pytorch

```
### Clone the repository and run the project
```bash
git clone https://github.com/xrai-lib/xai-texture.git
cd xai-texture
cd src
python main.py
```

## Usage
After completing the installation steps, you are ready to run the application. The main.py script is configured to demonstrate our methodology's application to breast cancer segmentation and explainability. The user interface will allow you to make use of all the functionalities available in the repository.

## Original Dataset
The original dataset used in this project is the Curated Breast Imaging Subset of DDSM (CBIS-DDSM), as described in the following paper:

- **Paper**: Lee RS, Gimenez F, Hoogi A, Miyake KK, Gorovoy M, Rubin DL. "A curated mammography data set for use in computer-aided detection and diagnosis research." *Sci Data*. 2017 Dec 19;4:170177. 
- **DOI**: [10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177)
- **PMID**: 29257132
- **PMCID**: PMC5735920

## Citation

MMSegmentation Contributors (2020) MMSegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark. https://github.com/open-mmlab/mmsegmentation
