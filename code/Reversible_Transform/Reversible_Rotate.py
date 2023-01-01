import cv2

from .Reversible_Transform import Reversible_Transform

class Reversible_Rotate(Reversible_Transform):
    def __init__(self, angle):
        self.angle = angle

    def forward(self, image):
        height, width = image.shape[:2] # image shape has 3 dimensions
        self.org_H, self.org_W = height, width
        
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_img = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
        self.new_h, self.new_w = rotated_img.shape[:2]
        return rotated_img

    def backward(self, image):
        height, width = image.shape[:2] # image shape has 3 dimensions
        
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, -1*self.angle, 1.)

        # find the new width and height bounds
        bound_w = int(self.org_W)
        bound_h = int(self.org_H)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        image = cv2.resize(image, (self.new_w, self.new_h))
        rotated_img = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
        return rotated_img