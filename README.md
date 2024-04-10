# Explainability in Deep Learning Segmentation Models for Breast Cancer by Analogy with Texture Analysis

This repository hosts the source code and related materials for the paper titled "Explainability in Deep Learning Segmentation Models for Breast Cancer by Analogy with Texture Analysis". Our project aims to advance the explainability of deep learning models in medical imaging, specifically in the context of breast cancer segmentation. By drawing analogies with texture analysis, we propose a novel approach to interpret the model's decisions, making these models more transparent and trustworthy for medical practitioners.

## Prerequisites

Before setting up the project, ensure you have the following installed on your system:
- Git
- Anaconda or Miniconda

## Installation

### Step 1: Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/xrai-lib/xai-texture.git
```

### Step 2: Create the Conda Environment

Navigate to the project's root directory:

```bash
cd xai-texture
```
Create a Conda environment named xai using the provided xai.yml file:
```bash
conda env create -f xai.yml
```
This command creates a Conda environment with all the necessary dependencies installed.

### Step 3: Activate the Environment
Activate the newly created environment using:
```bash
conda activate xai
```
### Step 4: Run the Application
Change to the src directory:
```bash
cd src
```
Execute the main application:
```bash
python main.py
```
## Usage
After completing the installation steps, you are ready to run the application. The main.py script is configured to demonstrate our methodology's application to breast cancer segmentation and explainability. Modify the script or create new ones as needed to explore the dataset and models further.
