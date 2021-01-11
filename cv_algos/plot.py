import matplotlib.pyplot as plt
import numpy as np


def draw_hist(img, filename, bins=256, figsize=(11, 7)):
    flatten_pixels = np.array(
        [img[r][c][z] for z in range(img.shape[2]) for c in range(img.shape[1]) for r in range(img.shape[0])])
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(flatten_pixels, bins=bins)
    ax.set_xlabel('Grey level')
    ax.set_ylabel('Count of pixels')
    plt.savefig(f'{filename}.jpg', dpi=300)