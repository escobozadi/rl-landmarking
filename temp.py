import cv2
import os
import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt


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

def vizualize(path, target):
    """
        ._.
    """
    np_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # (y: from up to down ,x)
    # transpose: (x: from right to left, y)
    pts = np.array([[0.5886666666666666, 0.3093869731800766],
                    [0.7573333333333332, 0.5114942528735632],
                    [0.2901135646687697, 0.7315857377433733]])
    width = np.array([0.2813333333333333, 0.16266666666666665, 0.3])
    height = np.array([0.1743295019157088, 0.17624521072796934, 0.1475095785440613])
    pts[:, 0] *= np_image.shape[1]
    pts[:, 1] *= np_image.shape[0]
    width *= np_image.shape[1]
    height *= np_image.shape[0]
    h, w = np_image.shape
    # for i in range(3):
    #     size = np.array([[(width[i])/2 + pts[i][0], (height[i])/2 + pts[i][1]]])
    #     pts = np.append(pts, size, axis=0)
    pts = np.floor(pts).astype(int)
    height = np.ceil(height).astype(int)
    width = np.ceil(width).astype(int)
    #print(np_image.flags)
    #print((np_image.T).flags)
    #np_image = np_image.transpose(1, 0)  # x,y
    #n = np.zeros((np_image.T).shape)
    #n[:] = (np_image.T)[:]
    np_image = cv2.transpose(np_image)
    print(np_image.shape)
    for i in range(3):
        np_image = cv2.circle(np_image, (pts[i][1], pts[i][0]), radius=10, color=255)
    # np_image = cv2.circle(np_image, (pts[0][0] + width[0], pts[0][1] - height[0]), radius=6, color=(0, 0, 255))
    cv2.imshow("image", np_image.transpose(1, 0))
    cv2.waitKey(0)

    return


if __name__ == '__main__':
    dir = "src/data/images/"
    dest = "src/data/norm_images/"
    im = "src/data/images/0a63a82d-4833___m4732_a4833_s4873_1_8_US_.png"
    lan = "src/data/landmarks/0a63a82d-4833___m4732_a4833_s4873_1_8_US_.txt"
    # imageNorm(path=dir, destination=dest)
    vizualize(im, lan)
    # landmarks = np.zeros((3, 2))
    # landmarks[:] = np.nan
    #
    # with open(lan) as t:
    #     lines = [x.strip() for x in list(t) if x]
    #     for l in lines:
    #         info = l.split(" ")
    #         id = int(info[0])
    #         print(info[1:])
    #         landmarks[id, :] = info[1:3]
    #         print(landmarks)
    #
    # landmarks = np.asarray(landmarks)
    # landmarks = landmarks.reshape((-1, landmarks.shape[1]))
    # print(landmarks)



