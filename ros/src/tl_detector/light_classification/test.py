from glob import glob

import os
from tl_classifier import TLClassifier
import cv2

classfier = TLClassifier()


def classify_images(images_folder):
    image_paths = glob(os.path.join(images_folder, '*.jpg'))
    for image_path in image_paths:
        test_image = cv2.imread(image_path)
        print(classfier.get_classification(test_image))


print('Red images')
classify_images('../images/red')

print('Green images')
classify_images('../images/green')
