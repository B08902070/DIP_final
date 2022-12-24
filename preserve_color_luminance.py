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

def main(content_dir, stylized_dir):
    content_dir = Path(content_dir)
    stylized_dir = Path(stylized_dir)

    content_names, content_imgs = get_images_and_names(content_dir)
    for ci in range(len(content_names)):
        for f in stylized_dir.glob('*'):
            if content_names[ci] in str(f):
                stylized_img = cv2.imread(f)
                new_img = luminance_only_transfer(content_img=content_imgs[ci], stylized_img=stylized_img)
                cv2.imwrite(f, new_img)




if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--content_dir", help="the path to the content dir", type=str)
    parser.add_argument("--stylized_dir", help="the path to the dir of stylized img", type=str)
    args = parser.parse_args()

    main(content_dir = args.content_dir, stylized_dir = args.stylized_dir)
