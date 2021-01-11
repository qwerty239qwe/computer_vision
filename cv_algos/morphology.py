from cv_algos.basics import binarize, downsample
import matplotlib.pyplot as plt
from cv_algos.kernels import *

J = ((1, 0), (0, 0), (0, -1))
K = ((-1, 0), (-1, 1), (0, 1))


def complement_2d(mat):
    return np.array([[255 - c for c in row] for row in mat], dtype='uint8')


def match_pix(kernel_corr, img_corr, pos):
    for corr in kernel_corr:
        abs_y = corr[1] + pos[1]
        if corr[0] + pos[0] not in img_corr:
            continue

        if abs_y in img_corr[corr[0] + pos[0]]:
            return True
    return False


def bin_dilation(img, kernel=BASIC_KERNEL):
    img_pix = {x: np.array([y for y, v in enumerate(img[x]) if v != 0]) for x in range(img.shape[0])}
    new_img = np.array([[255 if match_pix(kernel, img_pix, (i, j)) else 0 for j in range(img.shape[1])] for i in range(img.shape[0]) ], dtype='uint8')
    return new_img


def bin_erosion(img, kernel=BASIC_KERNEL):
    img_pix = {x: np.array([y for y, v in enumerate(img[x]) if v == 0]) for x in range(img.shape[0])}
    new_img = np.array([[0 if match_pix(kernel, img_pix, (i, j)) else 255 for j in range(img.shape[1])] for i in range(img.shape[0]) ], dtype='uint8')
    return new_img


def opening(img, kernel=BASIC_KERNEL):
    new_img = img.copy()
    new_img = bin_dilation(bin_erosion(new_img, kernel), kernel)
    return new_img


def closing(img, kernel=BASIC_KERNEL):
    new_img = img.copy()
    new_img = bin_erosion(bin_dilation(new_img, kernel), kernel)
    return new_img


def hit_and_miss(img, J=J, K=K):
    img_d = img.copy() # duplication
    img_c = complement_2d(img) # complement
    l_mat = bin_erosion(img_d, J)
    r_mat = bin_erosion(img_c, K)
    return np.array([[min(l_mat[i][j], r_mat[i][j]) for j in range(img.shape[1])] for i in range(img.shape[0]) ], dtype='uint8')


class ThinningOperator:
    def __init__(self, img):
        self.yok_img = self._to_yokoi_num(downsample(binarize(img, threshold=128), size=8))

    @staticmethod
    def _h_func(b, c, d, e):
        if b != c:
            return 0
        return 1 if (d != b) or (e != b) else 10

    def _f_func(self, block_arr):
        center = block_arr[1, 1]
        if center == 0:
            return 0

        hs = [self._h_func(center, block_arr[1, 2], block_arr[0, 2], block_arr[0, 1]),
              self._h_func(center, block_arr[0, 1], block_arr[0, 0], block_arr[1, 0]),
              self._h_func(center, block_arr[1, 0], block_arr[2, 0], block_arr[2, 1]),
              self._h_func(center, block_arr[2, 1], block_arr[2, 2], block_arr[1, 2])]
        return (sum(hs) % 10) + int(sum(hs) / 40) * 5

    def _to_yokoi_num(self, img):
        row, col = img.shape
        # add padding:
        pad_img = np.array(
            [[img[i][j] if (0 <= i < row) and (0 <= j < col) else -1 for i in range(-1, row + 1)] for j in
             range(-1, col + 1)])

        return np.array([[self._f_func(pad_img[i:i + 3, j:j + 3]) for i in range(row)] for j in range(col)], dtype=np.int8)

    def print_num_img(self, add_space=True):
        # for debugging
        for i in range(self.yok_img.shape[0]):
            for j in range(self.yok_img.shape[1]):
                print(self.yok_img[i, j], end=' ' if add_space else '') if self.yok_img[i, j] != 0 else print(' ',
                                                                                                    end=' ' if add_space else '')
            print()

    def draw_num_img(self, save_fig=True, **kwargs):
        fig, ax = plt.subplots(figsize=(16, 16), **kwargs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for i in range(self.yok_img.shape[0]):
            for j in range(self.yok_img.shape[1]):
                if self.yok_img[i, j] != 0:
                    ax.text((j + 0.4) / self.yok_img.shape[1], 1 - (i + 0.7) / self.yok_img.shape[0], self.yok_img[i, j])
        if save_fig:
            plt.savefig('Yokoi.jpg')
        plt.show()

    def _find_pair(self):
        pair_arr = np.ones(self.yok_img.shape) * -1
        for i in range(self.yok_img.shape[0]):
            for j in range(self.yok_img.shape[1]):
                if self.yok_img[i, j] != 1:
                    pair_arr[i, j] = 2 if self.yok_img[i, j] != -1 else 0
                    continue
                cond = [(i > 0 and self.yok_img[i - 1, j] == 1), (i < self.yok_img.shape[0] - 1 and self.yok_img[i + 1, j] == 1),
                        (j > 0 and self.yok_img[i, j - 1] == 1), (j < self.yok_img.shape[1] - 1 and self.yok_img[i, j + 1] == 1)]
                if any(cond):
                    pair_arr[i, j] = 1
                else:
                    pair_arr[i, j] = 2
        return pair_arr

    def get_thinning(self):
        pair_arr = self._find_pair()
        col, row = pair_arr.shape
        new_arr = np.array(
            [[pair_arr[j, i] if (0 <= i < col) and (0 <= j < row) else 0 for i in range(-1, col + 1)] for j in
             range(-1, row + 1)], dtype=np.int8)

        for j in range(1, 1 + row):
            for i in range(1, 1 + col):
                check_mat = new_arr[j - 1: j + 2, i - 1: i + 2]
                if new_arr[j, i] == 1 and self._f_func(
                        np.array([[1 if check_mat[y, x] != 0 else 0 for x in range(3)] for y in range(3)])) == 1:
                    new_arr[j, i] = 0
        return new_arr[1: 1 + col, 1: 1 + row]