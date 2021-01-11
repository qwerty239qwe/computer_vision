import cv2
import numpy as np


def binarize(img, threshold):
    new_img = np.array([[[0 if img[i, j, c] < threshold else 255 for c in range(img.shape[2])]
                         for j in range(int(img.shape[1]))]
                        for i in range(int(img.shape[0]))], dtype='uint8')
    return new_img


def hist_equalization(img):
    sum_pix = sum([1 for _ in range(img.shape[0]) for _ in range(img.shape[1])])
    p_i = np.zeros((256,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p_i[img[i][j]] += 1
    p_i = np.array([i / sum_pix for i in p_i])  # Probability
    cum_pi = np.array([sum(p_i[:i + 1]) for i in range(len(p_i))])  # CDF
    new_img = np.array([[int(cum_pi[img[i, j]] * 255) for j in range(img.shape[1])]
                                                              for i in range(img.shape[0])], dtype='uint8')

    return new_img


def downsample(img, size=8):
    return np.array([[img[size * j][size * i] for i in range(int(img.shape[1] / size))]for j in range(int(img.shape[0] / size))])