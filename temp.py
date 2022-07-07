import cv2
import os
import numpy as np


def imageNorm(path, destination):

    image_files = [f for f in os.listdir(path) if not f.startswith('.')]
    idx = 0
    for im in image_files:
        image = cv2.imread(path + im, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Image is empty: {}, {}".format(idx, im))
        im_norm = np.zeros(image.shape)
        im_norm = cv2.normalize(image, im_norm, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(destination + im, im_norm)
        idx += 1

    return


if __name__ == '__main__':
    dir = "src/data/images/"
    dest = "src/data/norm_images/"
    imageNorm(path=dir, destination=dest)

