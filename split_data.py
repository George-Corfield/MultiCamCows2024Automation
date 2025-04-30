from sklearn.model_selection import train_test_split
import os
import numpy as np


def split(folder_location):
    images = []
    date = folder_location.split("/")[-1]
    cameras = os.listdir(folder_location)
    for camera in cameras:
        classes = os.listdir(f'{folder_location}/{camera}')
        for i in classes:
            images.extend(
                f'{date}/{camera}/{i}/{j}' for j in os.listdir(f'{folder_location}/{camera}/{i}'))
    train, test = train_test_split(np.array(images), test_size=0.2)
    return train, test
