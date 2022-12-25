import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import cv2
""" 
the code is the implementation of the paper 'Preserving Color in Neural Artistic Style Transfer' by Gaytz et. al
the method used here is Luminance Approach metioned in the paper

"""
def get_images_and_names(image_dir):
    names = []
    imgs = []
    for p in image_dir.glob('*'):
        dir_sep = '\\' if '\\' in str(p) else '/'
        name = str(p).split(dir_sep)[-1].split('.')[0]
        names.append(name)

        img = cv2.imread(p)
        imgs.append(img)
    return names, imgs


def luminance_only_transfer(content_img, stylized_img):
    content_img=  cv2.resize(content_img, stylized_img.shape[:2])
    yuv_content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2YUV)
    yuv_stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2YUV)

    yuv_new_img = np.concatenate((yuv_stylized_img[:, :, :1], yuv_content_img[:, :, 1:]), axis=2)

    return cv2.cvtColor(yuv_new_img, cv2.COLOR_YUV2BGR)


