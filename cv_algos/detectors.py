import numpy as np


def eu_dist(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)


def opt_mirror(i_img, j_img, i_opt, j_opt):
    i_o, j_o = i_opt, j_opt
    if i_img <= i_o:
        i_o = i_img - 1
    elif i_o < 0:
        i_o = 0
    if j_img <= j_o:
        j_o = j_img - 1
    elif j_o < 0:
        j_o = 0
    return i_o, j_o


def get_bin(func):
    def wrap(img, cal, kernel, thres):
        f_img = func(img, cal, kernel)
        lower, higher = np.where(f_img > thres), np.where(f_img <= thres)
        f_img[lower], f_img[higher] = 0, 255
        return f_img.astype(np.uint8)

    return wrap


@get_bin
def opt(img, cal, kernel):
    img = img.copy().astype(np.float)
    i_img, j_img = img.shape
    f_img = np.array([[cal(
        [sum([pt[2] * img[opt_mirror(i_img, j_img, i + pt[1], j + pt[0])] for pt in k]) for k in kernel]) for j in
                       range(j_img)] for i in range(i_img)])
    return f_img


def p5_ind(i):
    return [i % 8 for i in range(i, i + 3)]


def n3_ind(i):
    return [i % 8 for i in range(i + 3, i + 8)]


def kernel_gen(kernel_tp):
    circu = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
    if kernel_tp == 'Kirsch compass':
        return [[(circu[k][0], circu[k][1], 5) for k in p5_ind(i)] + [(circu[k][0], circu[k][1], -3) for k in n3_ind(i)]
                for i in range(8)]
    elif kernel_tp == 'Robinson compass':
        return [
            [(circu[k][0], circu[k][1], v) for k, v in zip(p5_ind(i), (1, 2, 1))] + [(circu[k][0], circu[k][1], v) for
                                                                                     k, v in
                                                                                     zip(n3_ind(i)[1:4], (-1, -2, -1))]
            for i in range(8)]
    elif kernel_tp == 'Nevatia-Babu':
        vals = (100, 92, 78, 32, -32, -78, -92, -100)
        arrs = {d: np.zeros((5, 5)) for d in ['-90', '-60', '-30', '0', '30', '60']}

        arrs['60'][:, :2], arrs['60'][:, 3:] = 100, -100
        for d in ['-30', '0', '30']:
            arrs[d][:2, :], arrs[d][3:, :] = 100, -100
        for d in ['-90', '-60']:
            arrs[d][:, :2], arrs[d][:, 3:] = -100, 100

        # modify '60', '-60', '30', '-30'
        sp_ele = {'60': [(-2, 0), (-1, 0), (1, -1), (-2, 1), (2, -1), (-1, 1), (1, 0), (2, 0)],
                  '-60': [(-2, 0), (-1, 0), (1, 1), (-2, -1), (2, 1), (-1, -1), (1, 0), (2, 0)],
                  '30': [(0, -2), (0, -1), (-1, 1), (1, -2), (-1, 2), (1, -1), (0, 1), (0, 2)],
                  '-30': [(0, 2), (0, 1), (-1, -1), (1, 2), (-1, -2), (1, 1), (0, -1), (0, -2)]}
        for d, pos in sp_ele.items():
            for i, (x, y) in enumerate(pos):
                arrs[d][x + 2, y + 2] = vals[i]

        return [[(i - 2, j - 2, arr[j, i]) for i in range(5) for j in range(5)] for arr in arrs.values()]


"""
figs = {}

for opt_tp in ['robert', 'prewitt', 'sobel', 'FandC']:
    figs[opt_tp] = opt(img, eu_dist, kernel[opt_tp], THRESHOLD[opt_tp])
    
for opt_tp in ['Kirsch compass', 'Robinson compass', 'Nevatia-Babu']:
    figs[opt_tp] = opt(img, max, kernel_gen(opt_tp), THRESHOLD[opt_tp])
    
for i in ['robert', 'prewitt', 'sobel', 'FandC', 'Kirsch compass', 'Robinson compass', 'Nevatia-Babu']:
    cv2.imwrite(f'{i}.jpg', figs[i])
"""


def opt_mirror(i_img, j_img, i_opt, j_opt):
    i_o, j_o = i_opt, j_opt
    if i_img <= i_o:
        i_o = i_img - 1
    elif i_o < 0:
        i_o = 0
    if j_img <= j_o:
        j_o = j_img - 1
    elif j_o < 0:
        j_o = 0
    return i_o, j_o


def find_cross(mat, x, y):
    if mat[x, y] != 1:
        return 255
    for i in mat:
        for j in i:
            if j == -1:
                return 0
    return 255


def get_bin(func):
    def wrap(img, kernel, thres):
        f_img = func(img, kernel)
        lower, higher = np.where(f_img <= -thres), np.where(f_img >= thres)
        f2_img = np.zeros(f_img.shape)
        f2_img[lower], f2_img[higher] = -1, 1

        out_img = np.array(
            [[find_cross(f2_img[i - 1 if i != 0 else 0: i + 2 if i != f2_img.shape[0] - 1 else f2_img.shape[0],
                         j - 1 if j != 0 else 0: j + 2 if j != f2_img.shape[1] - 1 else f2_img.shape[0]],
                         1 if i != 0 else 0, 1 if j != 0 else 0)
              for j in range(f2_img.shape[1])] for i in range(f2_img.shape[0])])
        return out_img.astype(np.uint8)

    return wrap


def cal_LoG(x, y, sigma):
    K = -(x ** 2 + y ** 2) / (2 * (sigma ** 2))
    return -(1 + K) * (np.e ** K) / (np.pi * (sigma ** 4))


def cal_DoG(x, y, sigma_1, sigma_2):
    norm_1 = (np.e ** (-(x ** 2 + y ** 2) / (2 * sigma_1 ** 2))) / sigma_1
    norm_2 = (np.e ** (-(x ** 2 + y ** 2) / (2 * sigma_2 ** 2))) / sigma_2
    return (norm_1 - norm_2) / 2 * np.pi


def gen_kernel(size, sigma, scaling=1, add=0):
    s = int((size - 1) / 2)
    if isinstance(sigma, float):
        kernel_arr = (np.array(
            [[cal_LoG(x, y, sigma) for x in range(-s, s + 1)] for y in range(-s, s + 1)]) * scaling) + add
    elif len(sigma) == 2:
        kernel_arr = (np.array(
            [[cal_DoG(x, y, sigma[0], sigma[1]) for x in range(-s, s + 1)] for y in range(-s, s + 1)]) * scaling) + add

    return [(i - s, j - s, kernel_arr[i, j]) for i in range(size) for j in range(size)]