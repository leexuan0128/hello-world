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
        disparity = np.fromfile(pfm_file, endian + 'f') 

    new_disparity = np.reshape(disparity, newshape=(height, width))
    np.savetxt(r"./dispariy.txt", new_disparity, fmt='%f', delimiter=' ')

    disparity_img = np.reshape(disparity, newshape=(height, width, channels))
    #img = np.flipud(img).astype('uint8') 
    disparity_img = np.flipud(disparity_img).astype('float32')
    cv2.imwrite("./disparity_map.png", disparity_img)
    #show(img, "disparity")

    return disparity, [(height, width, channels), scale]

def create_depths_map(pfm_file_path):
    disparity, [(height, width, channels), scale] = read_pfm(pfm_file_path)
    focal_length = read_focal_length(15)[0]
    baseline = read_baseline('Transformation.txt')
    depth_map = (focal_length * baseline) / disparity

    new_depth = np.reshape(depth_map, newshape=(height, width))
    np.savetxt(r"./depth.txt", new_depth, fmt='%f', delimiter=' ')

    depth_img = np.reshape(depth_map, newshape=(height, width, channels))
    #depth_img = np.flipud(depth_map).astype('float32')
    depth_img = np.flipud(depth_img).astype('uint8')     
    cv2.imwrite("./depth_map.png", depth_img)

    #depth_map = np.squeeze(depth_img, 2)
    depth_map = new_depth[np.newaxis, np.newaxis, :, :]
    
    # depth_map (Nx1xHxW)
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
