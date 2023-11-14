import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import scipy as sp 
import scipy.ndimage as ndimage
import math
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, roc_curve, auc, accuracy_score, classification_report


import seaborn as sns
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB



def vignette_removal(image: np.ndarray, threshold: int = 50):
    '''
    Automatically crop the image to preserve the main content and eliminate any vignette.
    This process entails evaluating pixel values along the diagonal of the image.
    
    Args:
    - image (numpy ndarray): The input image to be cropped.
    - threshold (int): The threshold value to differentiate between the image and the vignette.
    
    Returns:
    - The coordinates of the cropping rectangle and the cropped image.
    '''
    # Find image dimensions
    height, width = image.shape[:2]
    greatest_common_divisor = np.gcd(height, width)

    # Calculate diagonal pixel coordinates
    y_coords = ([i for i in range(0, height, int(height / greatest_common_divisor))], [i for i in range(height - int(height / greatest_common_divisor), 0, -int(height / greatest_common_divisor))])
    x_coords = ([i for i in range(0, width, int(width / greatest_common_divisor))], [i for i in range(0, width, int(width / greatest_common_divisor))])

    # Compute mean pixel values along the diagonal
    coordinates = {'y1_1': 0, 'x1_1': 0, 'y2_1': height, 'x2_1': width, 'y1_2': height, 'x1_2': 0, 'y2_2': 0, 'x2_2': width}
    for i in range(2):
        diagonal_values = []
        y1_aux, x1_aux = 0, 0
        y2_aux, x2_aux = height, width
        for y, x in zip(y_coords[i], x_coords[i]):
            diagonal_values.append(np.mean(image[y, x, :]))

        # Determine the location of the first point where the threshold is crossed
        for idx, value in enumerate(diagonal_values):
            if value >= threshold and idx != 0:  # In the absence of a vignette, the value would be above the threshold at idx=0
                coordinates['y1_' + str(i + 1)] = y_coords[i][idx]
                coordinates['x1_' + str(i + 1)] = x_coords[i][idx]
                break

        # Find the location of the last point where the threshold is crossed
        for idx, value in enumerate(reversed(diagonal_values)):
            if value >= threshold and idx != 0:  # In the absence of a vignette, the value would be above the threshold at idx=0
                coordinates['y2_' + str(i + 1)] = y_coords[i][len(y_coords[i]) - idx]
                coordinates['x2_' + str(i + 1)] = x_coords[i][len(x_coords[i]) - idx]
                break

    # Define the coordinates for cropping the image
    y1 = max(coordinates['y1_1'], coordinates['y2_2'])
    y2 = min(coordinates['y2_1'], coordinates['y1_2'])
    x1 = max(coordinates['x1_1'], coordinates['x1_2'])
    x2 = min(coordinates['x2_1'], coordinates['x2_2'])

    cropped_image = image[y1:y2, x1:x2, :]

    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        cropped_image = image

    return cropped_image


def remove_hair(src: np.ndarray, se_size: int = 15):
    '''param : src --> Color image
               se_size --> Size of the structuring elements
      return : Inp --> Inpainted image with hair removed using an alternative method'''

    # Convert the original image to grayscale if it has more than one channel
    if len(src.shape) == 3:
        grayscale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = src

    # Structuring Element for morphological filtering
    se = cv2.getStructuringElement(1, (se_size, se_size))  # (17x17) '+' shaped SE
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))  # (17x17) square-shaped SE

    # Perform morphological blackHat filtering on the grayscale image to detect hair (and other objects) contours
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, se)
    blackhat2 = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, se2)
    enhanced_contours = blackhat + blackhat2

    # Threshold the enhanced contours to create a mask for inpainting
    ret, mask = cv2.threshold(enhanced_contours, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the original image using the mask
    Inpaint = cv2.inpaint(src, mask, 1, cv2.INPAINT_TELEA)

    return Inpaint


# Define the target size for resizing
target_size = (256, 256)
# Function to load and resize images
def load_and_resize_image(file_path):
    img = Image.open(file_path)
    img = img.resize(target_size)
    return img