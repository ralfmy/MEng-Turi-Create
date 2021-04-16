import os
import cv2
from PIL import Image, ImageOps, ImageEnhance

# AUGMENTATION
def apply_augmentation(src_dir, dest_dir):
    for entry in os.scandir(src_dir):
        if entry.path.endswith(".jpg"):
            image = Image.open(entry.path)
            enhancer = ImageEnhance.Brightness(image)
            brighter = enhancer.enhance(1.7)
            darker = enhancer.enhance(0.5)
            hflip = ImageOps.mirror(image)

            brighter.save(dest_dir + entry.name[:-4] + "_BRIGHTER.jpg")
            darker.save(dest_dir + entry.name[:-4] + "_DARKER.jpg")
            hflip.save(dest_dir + entry.name[:-4] + "_MIRRORED.jpg")

            image.save(dest_dir + entry.name)


# PREPROCESSING
def apply_preprocessing(src_dir, dest_dir):
    for entry in os.scandir(src_dir):
        if entry.path.endswith(".jpg"):
            image = cv2.imread(entry.path)
            # Apply Gaussian blur (de-noise, smoothing), and convert to greyscale
            blur = cv2.GaussianBlur(image, (5, 5), 0)
            grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
            image = grey
            cv2.imwrite(dest_dir + entry.name, image)