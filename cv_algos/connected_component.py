import cv2
import numpy as np
import pandas as pd


def get_perm_table(img_arr):
    # get perm table
    perm_table = np.ones((1, 4), dtype=np.int16) * -1
    row_record_table = np.ones((img_arr.shape[1], 2), dtype=np.int16) * - 1
    for r in range(img_arr.shape[0]):
        find_c = -1
        for c in range(img_arr.shape[1]):
            if img_arr[r][c][0] > 0 and find_c == -1:
                find_c = c
                if row_record_table[r][0] == -1:
                    row_record_table[r][0] = perm_table.shape[0] - 1
            elif img_arr[r][c][0] == 0 and find_c != -1:
                perm_table = np.vstack([perm_table, np.array([[r, find_c, c, - 1]], dtype=np.int16)])
                find_c = -1
        if find_c != -1:
            perm_table = np.vstack([perm_table, np.array([[r, find_c, img_arr.shape[1], -1]])])

        if row_record_table[r][0] != -1:
            row_record_table[r][1] = perm_table.shape[0] - 1

    return perm_table[1:, :], row_record_table


def label_empty_rows(perm_table, new_label, from_row, to_row):
    # label every empty(-1) entries of perm_table
    for i in range(from_row, to_row):
        if perm_table[i][3] == -1:
            perm_table[i][3] = new_label
            new_label += 1
    return new_label


def fill_all_row(perm_table, row_record_table, new_label):
    # fill perm_table's 'label field'
    for i in range(1, row_record_table.shape[0]):
        p_list, q_list = row_record_table[i - 1], row_record_table[i]
        p, q, p_last, q_last = p_list[0], q_list[0], p_list[1], q_list[1]
        while p < p_last and q < q_last:
            p_col_l, p_col_r = perm_table[p][1], perm_table[p][2] - 1
            q_col_l, q_col_r = perm_table[q][1], perm_table[q][2] - 1
            if p_col_l > q_col_r:
                if perm_table[q][3] == -1:
                    perm_table[q][3] = new_label
                    new_label += 1
                q += 1
            elif p_col_r < q_col_l:
                if perm_table[p][3] == -1:
                    perm_table[p][3] = new_label
                    new_label += 1
                p += 1
            else:
                if perm_table[q][3] != -1:
                    if perm_table[p][3] == -1:
                        raise ValueError('Element in p should be labeled first!')
                    assigned_lab = min(perm_table[q][3], perm_table[p][3])
                    for j in range(q):
                        if perm_table[j][3] == perm_table[p][3] or perm_table[j][3] == perm_table[q][3]:
                            perm_table[j][3] = assigned_lab

                    perm_table[q][3], perm_table[p][3] = assigned_lab, assigned_lab
                else:
                    perm_table[q][3] = perm_table[p][3]

                if p_col_r > q_col_r:
                    q += 1
                elif p_col_r < q_col_r:
                    p += 1
                else:
                    q, p = q + 1, p + 1

        if p == p_last:
            new_label = label_empty_rows(perm_table, new_label, q, q_last)

        if q == q_last:
            new_label = label_empty_rows(perm_table, new_label, p, p_last)

    return new_label


def get_connected_components_util(perm_table, new_label):
    # generate connected component dict
    connected_component = dict()

    for i, v in enumerate(perm_table):
        if v[3] == -1:
            perm_table[i][3] = new_label
            new_label += 1

        if v[3] in connected_component:
            connected_component[v[3]].append(i)
        else:
            connected_component[v[3]] = [i, ]

    return connected_component


def get_connected_components(perm_table, row_record_table):
    new_label = 0
    # assign labels to the first row
    new_label = label_empty_rows(perm_table, new_label, row_record_table[0][0], row_record_table[0][1])

    # assign labels to all of the others rows
    new_label = fill_all_row(perm_table, row_record_table, new_label)

    # second scanning and get all of the connected components
    connected_component = get_connected_components_util(perm_table, new_label)

    return connected_component


def draw_cross(img, pt_row, pt_col):
    # helper function that helps us to draw a cross symbol
    cv2.line(img, (pt_col, pt_row - 4), (pt_col, pt_row + 4), (0, 0, 255), 2)
    cv2.line(img, (pt_col - 4, pt_row), (pt_col + 4, pt_row), (0, 0, 255), 2)


def find_centroid(perm_table):
    # find the location of centroid by using a pandas dataframe
    row_top, row_bottom = min([perm_table.iloc[r, 0] for r in range(perm_table.shape[0])]), max(
        [perm_table.iloc[r, 0] for r in range(perm_table.shape[0])])
    row_pix = {r: sum([perm_table[perm_table[0] == r].iloc[i, 2] -
                       perm_table[perm_table[0] == r].iloc[i, 1] for i in
                       range(perm_table[perm_table[0] == r].shape[0])])
               for r in range(row_top, row_bottom + 1)}
    area = sum([r for r in row_pix.values()])

    col_leftest, col_rightest = min([perm_table.iloc[r, 1] for r in range(perm_table.shape[0])]), max(
        [perm_table.iloc[r, 2] for r in range(perm_table.shape[0])])
    col_pix = {
        c: sum([1 if perm_table.iloc[r, 1] <= c < perm_table.iloc[r, 2] else 0 for r in range(perm_table.shape[0])]) for
        c in range(col_leftest, col_rightest)}
    area_c = sum([c for c in col_pix.values()])

    if area_c != area:
        raise ValueError('Some error occured while the program was calculating area!')

    row_centroid = sum([i * r for i, r in row_pix.items()]) / area
    col_centroid = sum([i * c for i, c in col_pix.items()]) / area
    return row_centroid, col_centroid


def output_final_result(img, file_name):
    # the main function
    new_img = img.copy()
    perm_table, row_record_table = get_perm_table(new_img)
    connected_component = get_connected_components(perm_table, row_record_table)
    perm_df = pd.DataFrame(perm_table)
    for label, comp_ptr in connected_component.items():
        count_pix = sum([perm_table[c][2] - perm_table[c][1] for c in comp_ptr])
        if count_pix >= 500:
            row_ind = [perm_table[c][0] for c in comp_ptr]
            col_left_ind, col_right_ind = [perm_table[c][1] for c in comp_ptr], [perm_table[c][2] for c in comp_ptr]
            print('find a connected component, bounding box: ',
                  (min(row_ind), max(row_ind), min(col_left_ind), max(col_right_ind)))
            cv2.rectangle(new_img, (min(col_left_ind), min(row_ind),), (max(col_right_ind) - 1, max(row_ind),),
                          (0, 200, 0), 2)
            r_c, c_c = find_centroid(perm_df[perm_df[3] == label])
            draw_cross(new_img, int(r_c), int(c_c))
            print('centroid is located at', (r_c, c_c))
    cv2.imwrite(file_name, new_img)

