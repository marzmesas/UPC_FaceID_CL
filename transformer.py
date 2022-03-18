from torchvision import transforms
import os
from PIL import Image
import cv2
import numpy as np

transformed_path = './Database-with-pairs'
#Transformation to resize pictures
image_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([130, 130]), # Tune it
])
#Transformation to generate pairs
pair_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation=0.5, hue=0.5),
    transforms.GaussianBlur(15*15), # Tune it
    transforms.RandomHorizontalFlip(p=0.5)
])

#Applies the transformations to generate and save the pairs given a folder with 
#images (as a list)
def generate_pairs(folder): 
    if not os.path.exists(transformed_path):
        os.mkdir(transformed_path)
    for idx, image in enumerate(folder):
        image = np.array(image_transformations(image))
        pair = np.array(pair_transformations(image))
        cv2.imwrite(f'{transformed_path}/{idx:04}.png', image) #save image
        cv2.imwrite(f'{transformed_path}/{idx:04}_transformed.png', pair) #save pair
    

