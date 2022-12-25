from skimage.transform import swirl

from .Reversible_Transform import Reversible_Transform

class Reversible_Swirl(Reversible_Transform):
    def __init__(self, center, strength, radius):
        self.center = center
        self.strength = strength
        self.radius = radius

    def forward(self, image):
        new_img = (swirl(image, center = self.center, strength=self.strength, radius=self.radius)*255).astype(image.dtype)
        return new_img

    def backward(self, image):
        new_img = (swirl(image, center = self.center, strength=-1*self.strength, radius=self.radius)*255).astype(image.dtype)
        return new_img