from cv_algos.kernels import *


def pooling(kernel, img, pos, strategy='max'):
    max_x, max_y = img.shape
    res = np.array([img[kx + pos[0], ky + pos[1]] for kx, ky in kernel if (0 <= kx + pos[0] < max_x) and (0 <= ky + pos[1] < max_y)])
    return max(res) if strategy == 'max' else min(res)


def gs_dilation(img, kernel=BASIC_KERNEL):
    new_img = np.array([[pooling(kernel, img, (i, j), 'max') for j in range(img.shape[1])] for i in range(img.shape[0]) ], dtype='uint8')
    return new_img


def gs_erosion(img, kernel=BASIC_KERNEL):
    new_img = np.array([[pooling(kernel, img, (i, j), 'min') for j in range(img.shape[1])] for i in range(img.shape[0]) ], dtype='uint8')
    return new_img


def opening(img, kernel=BASIC_KERNEL):
    new_img = img.copy()
    new_img = gs_dilation(gs_erosion(new_img, kernel), kernel)
    return new_img


def closing(img, kernel=BASIC_KERNEL):
    new_img = img.copy()
    new_img = gs_erosion(gs_dilation(new_img, kernel), kernel)
    return new_img


def add_gaussian_noise(img, amp):
    return (img + np.random.normal(size=img.shape) * amp).astype(np.uint8)


def add_sp_noise(img, prob):
    new_img = img.copy()
    prob_mat = np.random.uniform(size=img.shape)
    new_img[np.where(prob_mat < prob)], new_img[np.where(prob_mat > 1-prob)] = 0 , 255
    return new_img


def box_filter(img, kernel_size):
    if kernel_size % 2 != 1:
        raise ValueError('please use an odd integer as the kernel size')

    expand = int((kernel_size - 1) / 2)
    gnt = range(-expand, expand + 1)
    row_filt = np.array([[np.mean([img[i + g, j] for g in gnt if i + g >= 0 and i + g < img.shape[0]]) for j in range(img.shape[1])] for i in range(img.shape[0])])
    col_filt = np.array([[np.mean([row_filt[i, j + g] for g in gnt if j + g >= 0 and j + g < row_filt.shape[1]]) for j in range(row_filt.shape[1])] for i in range(row_filt.shape[0])])

    return col_filt.astype(np.uint8)


def median_filter(img, kernel_size):
    if kernel_size % 2 != 1:
        raise ValueError('please use an odd integer as the kernel size')

    expand = int((kernel_size - 1) / 2)
    gnt = range(-expand, expand + 1)
    filtered_img = np.array([[np.median([img[i + g1, j + g2] for g1 in gnt if i + g1 >= 0 and i + g1 < img.shape[0]
                                                       for g2 in gnt if j + g2 >= 0 and j + g2 < img.shape[1]]) for j in range(img.shape[1])] for i in range(img.shape[0])])

    return filtered_img.astype(np.uint8)


def cal_SNR(img, bg_img):
    img, bg_img = img.copy() / 255, bg_img.copy() / 255
    noise = img.astype(np.float) - bg_img.astype(np.float)
    mu_s = np.sum(bg_img) / (bg_img.shape[0] * bg_img.shape[1])
    mu_n = np.sum(noise) / (noise.shape[0] * noise.shape[1])
    var_s = np.sum(np.array([[(bg_img[i, j] - mu_s) ** 2 for j in range(bg_img.shape[1])] for i in range(bg_img.shape[0])])) / (bg_img.shape[0] * bg_img.shape[1])
    var_n = np.sum(np.array([[(noise[i, j] - mu_n) ** 2 for j in range(noise.shape[1])] for i in range(noise.shape[0])])) / (noise.shape[0] * noise.shape[1])
    return 20 * np.log10(np.sqrt(var_s / var_n))


"""
noised = {'a1': add_gaussian_noise(img, 10), 'a2': add_gaussian_noise(img, 30), 
          'b1': add_sp_noise(img, .1), 'b2': add_sp_noise(img, .05), }
          
box_filtered = {key + f'_size_{s}': box_filter(img, s) for s in [3, 5] for key, img in noised.items()}
median_filtered = {key + f'_size_{s}': median_filter(img, s) for s in [3, 5] for key, img in noised.items()}
opn_cls = {key: closing(opening(img, kernel), kernel) for key, img in noised.items()}
cls_opn = {key: opening(closing(img, kernel), kernel) for key, img in noised.items()}

# calculate SNR
for name, dic in {'noise_added': noised, 'box_filtered': box_filtered, 'median_filtered': median_filtered, 'opened and closed': opn_cls, 'closed and opened':cls_opn}.items():
    for k, v in dic.items():
        print(f"{name} image, {k}'s SNR is: {cal_SNR(v, img)}")

# save images
for name, dic in {'noise_added': noised, 'box_filtered': box_filtered, 'median_filtered': median_filtered, 'opened and closed': opn_cls, 'closed and opened':cls_opn}.items():
    for k, v in dic.items():
        cv2.imwrite(f'{name}_{k}.jpg', v)

"""