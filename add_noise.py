import cv2
import numpy as np
from argparse import ArgumentParser

def modify_lightness_saturation(img):

    origin_img = img

    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    lightness = 0
    saturation = 300

    # lightness
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1

    # saturation
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1


    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))


def add_pattern(input_img, pattern_img):
    pattern_img = cv2.resize(pattern_img, (input_img.shape[1], input_img.shape[0]))
    new_img = input_img * 0.6 + pattern_img *0.4
    return new_img

def add_noise(input_img, method):
    pass





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", help='choose noise method among [uniform_noise, pattern_img]', type=str)
    parser.add_argument("--pattern_img", help='path to the pattern img if the method is pattern_img', 
                        default='./images/gray-oil-painting.jpg', type=str)
    parser.add_argument("--input_img", help='path to the input image', type=str)
    parser.add_argument("--out", help='output image path', type=str)

    args = parser.parse_args()

    input_img = cv2.imread(args.input_img)
    if args.method == 'pattern_img':
        pattern_img = cv2.imread(args.pattern_img)
        new_img = add_pattern(input_img, pattern_img)
    else:
        new_img = add_noise(input_img, args.method)

    cv2.imwrite(args.out, new_img) 
    