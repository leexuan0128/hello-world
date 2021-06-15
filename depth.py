from pathlib import Path
import numpy as np
import csv
import re
import cv2
import random
import math

from functional_data import *


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian  
        dispariy = np.fromfile(pfm_file, endian + 'f') 

    img = np.reshape(dispariy, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    #np.savetxt(r"./dispariy.txt", dispariy_true, fmt='%f', delimiter=' ')
    cv2.imwrite("./disparity_map_4.png", img)
    #show(img, "disparity")

    return dispariy, [(height, width, channels), scale]

def create_depths_map(pfm_file_path):
    disparity, [shape, scale] = read_pfm(pfm_file_path)
    focal_length = read_focal_length(15)[0]
    baseline = read_baseline('Transformation.txt')

    depth_map = (focal_length * baseline) / disparity
    depth_map = np.reshape(depth_map, newshape=shape)
    depth_map = np.flipud(depth_map).astype('uint8')
    #np.savetxt(r"./depth.txt", depth_map, fmt='%f', delimiter=' ')

    return depth_map

def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)

# def main():
#     data_path_dir = Path(r"./disparity/")
#     pfm_images = data_path_dir.joinpath('0048.pfm')
#     depths = create_depths_map(pfm_images)
#     show(depths, "depth_map")
#     cv2.imwrite("./depth_map_4.png", depths)

# if __name__ == '__main__':
#     main()
