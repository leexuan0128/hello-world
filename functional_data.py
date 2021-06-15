from pathlib import Path
import numpy as np
import csv
import re
import cv2
import random
import math

def read_transformation(transformation_path):
    with open(transformation_path, 'r') as f:
        lines = f.readlines()

        L_data = lines[0].split(' ')
        R_data = lines[1].split(' ')
        T_L = np.array(L_data, dtype=float).reshape(4,4)
        T_R = np.array(R_data, dtype=float).reshape(4,4)

    return T_L, T_R

def read_focal_length(focal_length):
    if focal_length == 15:
        K = [[450.0, 0.0, 479.5, 0.0],
            [0.0, 450.0, 269.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        fx = K[0][0]
        inv_K = np.linalg.inv(K)
    return fx, K, inv_K

    if focal_length == 35:
        K = [[1050.0, 0.0, 479.5, 0.0],
            [0.0, 1050.0, 269.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        fx = K[0][0]
        inv_K = np.linalg.inv(K)
    return fx, K, inv_K

def read_baseline(baseline_path):
    with open(baseline_path, 'r') as b:
        TL, TR = read_transformation('Transformation.txt')
        inv_TL = np.linalg.inv(TL)
        T_RL = np.dot(inv_TL, TR)
        print(T_RL)
        Baseline = round((T_RL[0][3]))

    return Baseline

# def main():
#     fx, K, inv_K = read_focal_length(15)
#     T = read_transformation('Transformation.txt')[0]
#     B = read_baseline('Transformation.txt')

# if __name__ == '__main__':
#     main()