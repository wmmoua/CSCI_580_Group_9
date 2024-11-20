import torch
import numpy as np
from PIL import Image
import os

def ProjectDataLoader():
    # change working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # get script's directory
    os.chdir(script_dir)
    images = []
    labels = []

    for filename in os.listdir("digits/"):
        image_path = os.path.join("digits/", filename)
        image = Image.open(image_path).convert("L")
        image_array = np.array(image)
        tensor = torch.tensor(image_array, dtype=torch.float32)
        normalized_tensor = (tensor - 127.5) / 127.5

        images.append(normalized_tensor)
        labels.append(filename[0])

    images = np.array(images)  # shape: (num_images, height, width)
    labels = np.array(labels)  # shape: (num_images)
    return images, labels

