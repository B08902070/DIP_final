import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import cv2
""" 
the code is the implementation of the paper 'Preserving Color in Neural Artistic Style Transfer' by Gaytz et. al
the method used here is Luminance Approach metioned in the paper

"""
def get_Y_mean_std(img):
    mean = np.mean(img[:, :, 0])
    std = np.std(img[:, :, 0])
    return mean, std

def Y_normalize(img, mean_content, var_content, mean_stylized, var_stylized):
    img[:, :, 0] = (img[:, :, 0] - mean_stylized)/var_stylized  * var_content + mean_content
    return img

def luminance_only_transfer(content_img, stylized_img):
    content_img = np.array(content_img, dtype=np.float32)
    stylized_img = np.array(stylized_img, dtype=np.float32)

    content_img=  cv2.resize(content_img, (stylized_img.shape[1], stylized_img.shape[0]))
    yuv_content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2YUV)
    yuv_stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2YUV)

    mean_content, std_content = get_Y_mean_std(yuv_content_img)
    mean_stylized, std_stylized = get_Y_mean_std(yuv_stylized_img)


    yuv_new_img = np.concatenate((yuv_stylized_img[:, :, :1], yuv_content_img[:, :, 1:]), axis=2)
    yuv_new_img = Y_normalize(yuv_new_img, mean_content, std_content, mean_stylized, std_stylized)

    return cv2.cvtColor(yuv_new_img, cv2.COLOR_YUV2BGR)


