import os

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def centroid(img, lcc=False):
    if lcc:
        img = img.astype(np.uint8)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(
            img, connectivity=4)
        sizes = stats[:, -1]
        if len(sizes) > 2:
            max_label = 1
            max_size = sizes[1]

            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]

            img2 = np.zeros(output.shape)
            img2[output == max_label] = 255
            img = img2

    if len(img.shape) > 2:
        M = cv.moments(img[:, :, 1])
    else:
        M = cv.moments(img)

    if M["m00"] == 0:
        return (img.shape[0] // 2, img.shape[1] // 2)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def to_polar(input_img, center):
    input_img = input_img.astype(np.float32)
    value = np.sqrt(((input_img.shape[0] / 2.0)**2.0) +
                    ((input_img.shape[1] / 2.0)**2.0))
    polar_image = cv.linearPolar(input_img, center, value,
                                 cv.WARP_FILL_OUTLIERS)
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    return polar_image


def to_cart(input_img, center):
    input_img = input_img.astype(np.float32)
    input_img = cv.rotate(input_img, cv.ROTATE_90_CLOCKWISE)
    value = np.sqrt(((input_img.shape[1] / 2.0)**2.0) +
                    ((input_img.shape[0] / 2.0)**2.0))
    polar_image = cv.linearPolar(input_img, center, value,
                                 cv.WARP_FILL_OUTLIERS + cv.WARP_INVERSE_MAP)
    polar_image = polar_image.astype(np.uint8)
    return polar_image


if __name__ == "__main__":
    image = cv.imread('test_images/30.tif')
    plt.imshow(image)

    center = centroid(image)
    plt.scatter(center[0], center[1])
    plt.show()

    polar = to_polar(image, center)
    plt.imshow(polar)
    plt.show()

    cart = to_cart(polar, center)
    plt.imshow(cart)
    plt.show()
