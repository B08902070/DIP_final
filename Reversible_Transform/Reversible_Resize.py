import cv2

from .Reversible_Transform import Reversible_Transform


class Reversible_Resize(Reversible_Transform):
    def __init__(self, size1, size2=None):
        self.size1 = size1
        self.size2 = size2 if size2 is not None else size1

    def forward(self, image):
        self.org_H, self.org_W = image.shape[:2]
        new_img = cv2.resize(image, (self.size1, self.size2))
        return new_img

    def backward(self, image):
        new_img = cv2.resize(image, (self.org_H, self.org_W))
        return new_img