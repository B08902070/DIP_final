import imageio
from pathlib import Path
import numpy as np
import cv2

def get_images_and_names(image_dir):
    names = []
    imgs = []
    for p in image_dir.glob('*'):
        dir_sep = '\\' if '\\' in str(p) else '/'
        name = str(p).split(dir_sep)[-1].split('.')[0]
        names.append(name)

        img = imageio.imread(p)
        imgs.append(img)
    return names, imgs


def RGB2YUV(rgb_img):
    rgb_img = np.array(rgb_img)
    org_shape = rgb_img.shape
    rgb_img = np.reshape(rgb_img, (-1, 3))

    mat = np.array([[0.299, -0.1687, 0.5], 
                    [0.587, -0.3313, 0.5],
                    [0.114, -0.4187, -0.0813]])

    yuv_img = np.matmul(rgb_img, mat)
    yuv_img = yuv_img + np.broadcast_to(np.array([0, 128, 128]), yuv_img.shape)
    yuv_img = np.reshape(yuv_img, org_shape)

    return yuv_img

def YUV2RGB(yuv_img):
    yuv_img = np.array(yuv_img)
    org_shape = yuv_img.shape
    yuv_img = np.reshape(yuv_img, (-1, 3))

    mat = np.array([[1, 1, 1], 
                    [0, -0.34414, 1.772],
                    [1.402, -0.71414, 0]])

    rgb_img = np.matmul(yuv_img, mat)
    rgb_img = rgb_img + np.broadcast_to(np.array([1.402*(-128), (0.34414+0.71414)*128, -1.772*128]), rgb_img.shape)
    rgb_img = np.reshape(rgb_img, org_shape)

    return rgb_img

def luminance_only_transfer(content_img, stylized_img):
    content_img=  cv2.resize(content_img, stylized_img.shape[:2])
    yuv_content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2YUV)
    yuv_stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2YUV)

    yuv_new_img = np.concatenate((yuv_stylized_img[:, :, :1], yuv_content_img[:, :, 1:]), axis=2)

    return cv2.cvtColor(yuv_new_img, cv2.COLOR_YUV2BGR)


content_img = cv2.imread(str(Path('LinearStyleTransfer/data/content/1.jpg')))
#new_content = YUV2RGB(RGB2YUV(content_img))
#imageio.imwrite("new_content.jpg", new_content)
stylized_img = cv2.imread(str(Path('LinearStyleTransfer/Output/1_3314.png')))

new_img = luminance_only_transfer(content_img=content_img, stylized_img=stylized_img)
cv2.imwrite("color_transfer_luminance.jpg", new_img)