import cv2
import numpy as np
from .Reversible_Transform import Reversible_Transform


class Reversible_Resize(Reversible_Transform):
    def __init__(self, size1, size2=None):
        self.size1 = size1
        self.size2 = size2 if size2 is not None else size1

    def forward(self, image):
        self.org_H, self.org_W = image.shape[:2]
        new_img = cv2.resize(image, (self.size1, self.size2))
        blank_img = np.zeros(image.shape, dtype=np.uint8)
        blank_img[:self.size1, :self.size2, :] = new_img.copy()
        return blank_img

    def backward(self, image):
        new_img = cv2.resize(image, (self.org_H, self.org_W))
        new_img = new_img[:self.size1, :self.size2, :].copy()
        new_img = cv2.resize(new_img, (self.org_H, self.org_W))
        return new_img