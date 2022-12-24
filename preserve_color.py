import torch
from argparse import ArgumentParser
from pathlib import Path
import imageio
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

        img = imageio.imread(p)
        imgs.append(img)
    return names, imgs
        
def color_transfer(content_img, style_img):
    pass

def main(content_dir, style_dir, save_dir):
    content_dir = Path(content_dir)
    style_dir = Path(style_dir)

    """get names"""
    content_names, content_imgs = get_images_and_names(content_dir)
    style_names, style_imgs = get_images_and_names(style_dir)
    
    for i in range(len())
    for f in 




if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--content_dir", help="the path to the content dir", type=str)
    parser.add_argument("--style_dir", help="path to the style dir", type=str)
    parser.add_argument("--save_dir", help="path to the new style dir after color preserve operation", type=str)
    args = parser.parse_args()

    main(content_dir=args.content_dir, style_dir=args.style_dir, save_dir=args.save_dir)