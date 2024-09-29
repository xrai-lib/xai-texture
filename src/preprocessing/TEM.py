import cv2
import os
import numpy as np
import config

# Function to apply convolution with a given kernel
def apply_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def generate_TEM_dataset(test):

    # Define the convolution kernels
    L5 = np.array([1, 4, 6, 4, 1]).reshape(1, 5)  # L5 mask Level
    E5 = np.array([-1, -2, 0, 2, 1]).reshape(1, 5)  # E5 mask Edge
    S5 = np.array([-1, 0, 2, 0, -1]).reshape(1, 5)  # S5 mask Spot
    R5 = np.array([1, -4, 6, -4, 1]).reshape(1, 5)  # R5 mask Ripple

    # Directory paths
    if test:
        print("Generating TEM feature test dataset")
        input_images_dir = config.patch_dataset_path + '/test/images'
        output_dir = config.TEM_dataset_path + '/test/textures'
    else:
        print("Generating TEM feature train dataset")
        input_images_dir = config.patch_dataset_path + '/train/images'
        output_dir = config.TEM_dataset_path + '/train/textures'

    # Iterate over images in the input folder
    for filename in os.listdir(input_images_dir):
        # Load grayscale image
        image = cv2.imread(os.path.join(input_images_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Calculate texture features
        L5E5 = apply_convolution(image, np.outer(L5, E5))
        E5L5 = apply_convolution(image, np.outer(E5, L5))
        feature_1 = (L5E5 + E5L5) / 2

        L5S5 = apply_convolution(image, np.outer(L5, S5))
        S5L5 = apply_convolution(image, np.outer(S5, L5))
        feature_2 = (L5S5 + S5L5) / 2

        L5R5 = apply_convolution(image, np.outer(L5, R5))
        R5L5 = apply_convolution(image, np.outer(R5, L5))
        feature_3 = (L5R5 + R5L5) / 2

        E5S5 = apply_convolution(image, np.outer(E5, S5))
        S5E5 = apply_convolution(image, np.outer(S5, E5))
        feature_4 = (E5S5 + S5E5) / 2

        E5R5 = apply_convolution(image, np.outer(E5, R5))
        R5E5 = apply_convolution(image, np.outer(R5, E5))
        feature_5 = (E5R5 + R5E5) / 2

        R5S5 = apply_convolution(image, np.outer(R5, S5))
        S5R5 = apply_convolution(image, np.outer(S5, R5))
        feature_6 = (R5S5 + S5R5) / 2

        feature_7 = apply_convolution(image, np.outer(S5, S5))  # S5S5

        feature_8 = apply_convolution(image, np.outer(E5, E5))  # E5E5

        feature_9 = apply_convolution(image, np.outer(R5, R5))  # R5R5

        # Create folder for texture features if it doesn't exist
        for idx, feature in enumerate([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9], start=1):
            feature_folder = os.path.join(output_dir, f'Feature_{idx}')
            os.makedirs(feature_folder, exist_ok=True)
            output_image_path = os.path.join(feature_folder, filename)
            cv2.imwrite(output_image_path, feature)

    print("Texture extraction completed.")
