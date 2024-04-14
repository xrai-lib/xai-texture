# Results

## Model Evaluation

There are three evaluation files:

- `Deeplab_test.csv`
- `FCN_test.csv`
- `UNET_test.csv`

These files contain the Intersection over Union (IOU) scores of different models trained on 10 different datasets: CBIS-DDSM-Patches and nine different CBIS-DDSM-TEM-Features datasets.

IOU measures the overlap between the predicted segmentation mask and the ground truth mask. It is calculated as the ratio of the intersection of the predicted and ground truth masks to their union. Pixel accuracy measures the proportion of correctly classified pixels in the segmentation output compared to the ground truth.


## GLCM results

The GLCM value is calculated for last layer of each model. The results for each model is saved in csv files for corresponding models inside results/GLCM folder location.


### GLCM Folder

The `GLCM` folder contains CSV files that provide average absolute differences for 13 different GLCM properties. Lower values indicate less difference and more similarity.

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

## LTEM results

The Feature_10 dataset which is the model trained with original dataset is compared with model trained with each feature dataset using cosine similarity function. The results are saved in csv files for corresponding models inside results/LTEM_Cosine_Similarity folder location.

## LTEM_Cosine_Similarities Folder

The `LTEM_Cosine_Similarities` folder contains CSV files that include cosine similarities of different Laws mask texture energy measures.

### Model Evaluation Files

Additionally, there are three evaluation files:

- `Deeplab_test.csv`
- `FCN_test.csv`
- `UNET_test.csv`

These files contain the Intersection over Union (IOU) scores of different models trained on 10 different datasets: CBIS-DDSM-Patches and nine different CBIS-DDSM-TEM-Features datasets.
