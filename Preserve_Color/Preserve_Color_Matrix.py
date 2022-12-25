import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import cv2
""" 
the code is the implementation of the paper 'Preserving Color in Neural Artistic Style Transfer' by Gaytz et. al
the method used here is Image Analogies metioned in the paper

"""

def get_images_and_names(image_dir):
    names = []
    imgs = []
    for p in image_dir.glob('*'):
        dir_sep = '\\' if '\\' in str(p) else '/'
        name = str(p).split(dir_sep)[-1].split('.')[0]
        names.append(name)

        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return names, imgs


def cal_img_mean_covar(img):
    img = np.array(img, dtype=np.float32)
    mu = np.mean(img, axis=(0, 1))

    tmp_mat = np.reshape(img, (-1, 3))
    tmp_mat = tmp_mat - np.broadcast_to(mu, (tmp_mat.shape))
    covar = np.cov(tmp_mat.T)

    return np.expand_dims(mu, axis=1), covar

def cal_half_covar(covar):
    Lambda, U = np.linalg.eig(covar)
    Lambda = np.diag(Lambda**(1/2))

    half_covar = np.matmul(U, Lambda)
    half_covar = np.matmul(half_covar, U.T)
    return half_covar


def color_transfer(content_img, style_img):
    mean_content, covar_content = cal_img_mean_covar(content_img)
    mean_style, covar_style = cal_img_mean_covar(style_img)

    half_covar_content = cal_half_covar(covar_content)
    half_covar_style = cal_half_covar(covar_style)

    neg_half_covar_style = np.linalg.inv(half_covar_style)

    A = np.matmul(half_covar_content, neg_half_covar_style)
    b = mean_content - np.matmul(A, mean_style)

    org_shape = style_img.shape
    style_img = np.reshape(np.array(style_img), (-1, 3))
    style_img = np.moveaxis(style_img, -1, 0)
    
    new_img = np.matmul(A, style_img) + np.broadcast_to(b, style_img.shape)
    new_img = np.moveaxis(new_img, 0, -1)
    new_img = np.reshape(new_img, org_shape)

    return new_img 






