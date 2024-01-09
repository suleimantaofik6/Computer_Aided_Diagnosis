# CADx PROJECT: SKIN LESION CLASSIFICATION CHALLENGE USING CLASSICAL MACHINE LEARNING APPROACH 
This repository contain the code for the classification of skin lesions through a classical machine-learning (ML) approach employing a range of ML classifier models. The approach addresses both two-class and three-class problems, improving classification accuracy despite huge class imbalances and lesion variations.

## DATASET
The challenge dataset includes images from the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), [BCN_20000](https://paperswithcode.com/dataset/bcn-20000), and [MSK_datasets](https://paperswithcode.com/dataset/msk), offering a rich variety of dermatological samples for analysis.

## Project Division
This project is in two parts:
* Binary classification
* Multiclass classification

## Binary Classification Pipeline
The folder `Binary` contains the codes for the binary aspect of this project. The task is to classify 15,000 skin lession images into benign or others. It is a balanced dataset and the code is structured into three pipeline:

### 1. Data Preprocessing: Vignette Frame and Hair Removal
The `Binary/Preprocessing and Feature extraction/skin_lesion_extraction.py` script conatains two functions needed for vignette frame (cropped) and Hair removal
* vignette_removal function: Takes an image and a threshold as input parameters. It then detects the threshold where darkening occurs and crops the image accordingly.
* remove_hair function: Accept two input parameters i.e. an image and a structuring element. The function applies morphological black-hat filtering and thresholding operations to remove black hairs.

### 2. Feature Extraction
The `Binary/Preprocessing and Feature extraction/skin_lesion_feature_extraction.py` script conatains severaal defined functions for feature extration. These features are extracted using the ABCD rule, encompassing image `asymmetry,` `border` characteristics, `color` properties, and image `diameter.` These features are:
* Intensity and Color Features: `grayscale intensity, rgb, and hsv`
* Shape Features: `perimeter, area, circularity, compactness, and hu moments`
* Gray Level Co-Occurrence Matrix: `energy, correlation, dissimilarity, homogeneity, contrast, and Angular Second Moment (ASM)`
* Local Binary Pattern: `histogram of lbp values`

### 3. Machine Learning


## Multiclass classification
