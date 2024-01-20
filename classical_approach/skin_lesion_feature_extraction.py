from skin_lesion_preprocessing import *

from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from skimage import color

def extract_intensity_and_color_features(original_roi):
    # Convert the image to a NumPy array
    original_roi_array = np.array(original_roi)

    # Ensure that the image is in RGB format
    if original_roi_array.ndim == 2:
        original_roi_array = color.gray2rgb(original_roi_array)

    # Calculate intensity features
    mean_intensity = np.mean(original_roi_array)
    std_intensity = np.std(original_roi_array)
    max_intensity = np.max(original_roi_array)
    min_intensity = np.min(original_roi_array)

    # Separate channels
    r_channel, g_channel, b_channel = cv2.split(original_roi_array)

    # Calculate color features in RGB
    mean_r, mean_g, mean_b = np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)
    std_r, std_g, std_b = np.std(r_channel), np.std(g_channel), np.std(b_channel)
    max_r, max_g, max_b = np.max(r_channel), np.max(g_channel), np.max(b_channel)
    min_r, min_g, min_b = np.min(r_channel), np.min(g_channel), np.min(b_channel)

    # Convert the image to HSV
    hsv_image = color.rgb2hsv(original_roi_array)

    # Extract color statistics
    hsv_mean = np.mean(hsv_image, axis=(0, 1))
    hsv_std = np.std(hsv_image, axis=(0, 1))
    hsv_min = np.min(hsv_image, axis=(0, 1))
    hsv_max = np.max(hsv_image, axis=(0, 1))
    hsv_skewness = skew(hsv_image, axis=(0, 1))
    hsv_kurt = kurtosis(hsv_image, axis=(0, 1))
    hsv_entropy = shannon_entropy(hsv_image, base=2)

    rgb_mean = np.mean(original_roi_array, axis=(0, 1))
    rgb_std = np.std(original_roi_array, axis=(0, 1))
    rgb_min = np.min(original_roi_array, axis=(0, 1))
    rgb_max = np.max(original_roi_array, axis=(0, 1))
    rgb_skewness = skew(original_roi_array, axis=(0, 1))
    rgb_kurt = kurtosis(original_roi_array, axis=(0, 1))
    rgb_entropy = shannon_entropy(original_roi_array, base=2)

    features = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'max_intensity': max_intensity,
        'min_intensity': min_intensity,
        'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
        'std_r': std_r, 'std_g': std_g, 'std_b': std_b,
        'max_r': max_r, 'max_g': max_g, 'max_b': max_b,
        'min_r': min_r, 'min_g': min_g, 'min_b': min_b,
        'hsv_mean_h': hsv_mean[0], 'hsv_mean_s': hsv_mean[1], 'hsv_mean_v': hsv_mean[2],
        'hsv_std_h': hsv_std[0], 'hsv_std_s': hsv_std[1], 'hsv_std_v': hsv_std[2],
        'hsv_max_h': hsv_max[0], 'hsv_max_s': hsv_max[1], 'hsv_max_v': hsv_max[2],
        'hsv_min_h': hsv_min[0], 'hsv_min_s': hsv_min[1], 'hsv_min_v': hsv_min[2],
        'hsv_skewness_h': hsv_skewness[0], 'hsv_skewness_s': hsv_skewness[1], 'hsv_skewness_v': hsv_skewness[2],
        'hsv_kurt_h': hsv_kurt[0], 'hsv_kurt_s': hsv_kurt[1], 'hsv_kurt_v': hsv_kurt[2],
        'hsv_entropy': hsv_entropy,
        'rgb_mean_r': rgb_mean[0], 'rgb_mean_g': rgb_mean[1], 'rgb_mean_b': rgb_mean[2],
        'rgb_std_r': rgb_std[0], 'rgb_std_g': rgb_std[1], 'rgb_std_b': rgb_std[2],
        'rgb_max_r': rgb_max[0], 'rgb_max_g': rgb_max[1], 'rgb_max_b': rgb_max[2],
        'rgb_min_r': rgb_min[0], 'rgb_min_g': rgb_min[1], 'rgb_min_b': rgb_min[2],
        'rgb_skewness_r': rgb_skewness[0], 'rgb_skewness_g': rgb_skewness[1], 'rgb_skewness_b': rgb_skewness[2],
        'rgb_kurt_r': rgb_kurt[0], 'rgb_kurt_g': rgb_kurt[1], 'rgb_kurt_b': rgb_kurt[2],
        'rgb_entropy': rgb_entropy,
    }

    return features

def extract_shape_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours using Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour by area
    contour = max(contours, key=cv2.contourArea) if contours else np.array([[0, 0]])

    # Calculate shape features
    perimeter = cv2.arcLength(contour, closed=True)
    area = cv2.contourArea(contour)

    # Calculate circularity and compactness
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0

    # Calculate Hu moments
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Create a dictionary to store shape features
    shape_features = {
        'perimeter': perimeter,
        'area': area,
        'circularity': circularity,
        'compactness': compactness,
        'hu_moments_1': hu_moments[0],
        'hu_moments_2': hu_moments[1],
        'hu_moments_3': hu_moments[2],
        'hu_moments_4': hu_moments[3],
        'hu_moments_5': hu_moments[4],
        'hu_moments_6': hu_moments[5],
        'hu_moments_7': hu_moments[6]
    }

    return shape_features

# Function to extract Gray-Level Co-occurrence Matrix (GLCM) features
def extract_glcm_features(original_roi):
    # Convert 'original_roi' to a NumPy array if it's an 'Image' object
    if isinstance(original_roi, Image.Image):
        original_roi = np.array(original_roi)

    # Ensure that the image is in grayscale
    if original_roi.ndim == 3:
        original_roi = cv2.cvtColor(original_roi, cv2.COLOR_RGB2GRAY)
        
    glcm_features = pd.DataFrame()
    params = [(1, 0), (2, 0), (5, 0), (12, 0),
              (1, np.pi/6), (2, np.pi/6), (5, np.pi/6), (12, np.pi/6),
              (1, np.pi/4), (2, np.pi/4), (5, np.pi/4), (12, np.pi/4),
              (1, np.pi/2), (2, np.pi/2), (5, np.pi/2), (12, np.pi/2),]
    for i, (d, a) in enumerate(params):
        GLCM = graycomatrix(original_roi, [d], [a], symmetric=True, normed=True)
        GLCM_Energy = graycoprops(GLCM, 'energy')[0]
        glcm_features['Energy'+str(i+1)] = GLCM_Energy
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        glcm_features['Correlation'+str(i+1)] = GLCM_corr
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        glcm_features['Dissimilarity'+str(i+1)] = GLCM_diss
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        glcm_features['Homogeneity'+str(i+1)] = GLCM_hom
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]
        glcm_features['Contrast'+str(i+1)] = GLCM_contr
        GLCM_asm = graycoprops(GLCM, 'ASM')[0]
        glcm_features['ASM'+str(i+1)] = GLCM_asm
    return glcm_features


def extract_lbp_features(image, num_points=8, radius=1, method='uniform'):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the image to grayscale if it's not already
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Extract LBP features
    lbp = local_binary_pattern(image_array, P=num_points, R=radius, method=method)

    # Calculate histogram of LBP features
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Create a dictionary to store LBP features
    lbp_features = {"lbp_" + str(i): hist[i] for i in range(len(hist))}

    return lbp_features