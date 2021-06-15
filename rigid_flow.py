import sys
sys.path.append(r"./reprojections")

from backprojection import Backprojection
from transformation3d import Transformation3D
from projection import Projection
from reprojection import Reprojection

from depth import*
from functional_data import*

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# from raft import RAFT
# from utils import flow_viz
# from utils.utils import InputPadder

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)
    
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.
    
    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：
"""

def rigid_flow():
    # width = 960
    # height = 540
    fx, K, inv_K = read_focal_length(15)
    T = read_transformation('Transformation.txt')[0]
    B = read_baseline('Transformation.txt')

    disparity_path_dir = Path(r"./disparity/")
    pfm_images = disparity_path_dir.joinpath('0048.pfm')
    depth = create_depths_map(pfm_images)

    reprojections = Reprojection(540,960)
    reprojections.forward(depth, T, K, inv_K)


if __name__ == '__main__':
    rigid_flow()
