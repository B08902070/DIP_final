import cv2
import numpy as np

def get_Y_mean_std(img):
    mean = np.mean(img[:, :, 0])
    std = np.std(img[:, :, 0])
    return mean, std

def Y_normalize(img, mean, std, mean_target, std_target):
    img[:, :, 0] = (img[:, :, 0] - mean)/std  * std_target + mean_target
    return img

def add_noise(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y_mean_img, Y_std_img = get_Y_mean_std(img)


    noise_img = cv2.imread("images/noise.jpg")
    noise_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    noise_img = cv2.resize(noise_img, (img.shape[1], img.shape[0]))
    Y_mean_noise, Y_std_noise = get_Y_mean_std(noise_img)

    noise_img = Y_normalize(noise_img, Y_mean_noise, Y_std_noise, Y_mean_img, Y_std_img)

    new_img = img * 0.6 + noise_img *0.4
    return new_img