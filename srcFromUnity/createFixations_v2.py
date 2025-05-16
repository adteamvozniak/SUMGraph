import sys
from socket import herror
from multiprocessing.pool import ThreadPool as Pool
import cv2
import math
import os
import os.path as path
import numpy as np
import scipy.io as io
memory = 2
from scipy import optimize
import matplotlib.pyplot as plt
import json
from settings import visible_fov
fixation_image = []
from tqdm import tqdm
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def grayToRgb(img):
    gray = cv2.imread(img, 1)
    red = np.zeros(gray.shape,np.float32)
    for i in range(gray.shape[0]):
        j = 0
        for j in range(gray.shape[1]):
            val = gray[i, j, 0]
            gb = 255 - val
            red[i, j] = [255, gb, gb]

    red = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
    return red


def increaseSal_3(img_resolution, name, fixation_point, dva1, dva2):
    Xin, Yin = np.mgrid[0:img_resolution[0], 0:img_resolution[1]]
    color_map = plt.cm.get_cmap('binary')
    reversed_color_map = color_map.reversed()
    blank_image = np.zeros((img_resolution[0], img_resolution[1]), np.uint8)
    gazeGauss = (gaussian(3, fixation_point["x"], fixation_point["y"], dva1, dva2))(Xin, Yin)
    blank_image[round(fixation_point["x"]), round(fixation_point["y"])] = 1
    plt.imsave(name, gazeGauss, cmap=reversed_color_map)

    head, tail = os.path.split(name)
    tail = tail.replace(".png", "")
    new_name_mat = name.replace("png", "mat")
    io.savemat(new_name_mat, mdict={"fixations": blank_image, "name":tail, "resolution":np.array(([img_resolution[0],img_resolution[1]])), "gaze":np.array((round(fixation_point["x"]), round(fixation_point["y"])))})

