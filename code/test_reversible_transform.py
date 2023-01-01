import os
import cv2
from Reversible_Transform.Reversible_Rotate import Reversible_Rotate
from Reversible_Transform.Reversible_Resize import Reversible_Resize
from Reversible_Transform.Reversible_Swirl import Reversible_Swirl
from Reversible_Transform.Reversible_Wavy import Reversible_Wavy

output_dir = "./test_reverse_output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread("images/chicago.jpg")
H, W = img.shape[:2]


"""Test Reversible Rotate"""
rot = Reversible_Rotate(90)
rotated_img = rot.forward(img)
cv2.imwrite(output_dir + "rotate_90_img.jpg", rotated_img)
recover_img = rot.backward(rotated_img)
cv2.imwrite(output_dir + "recover_rotate_90.jpg", recover_img)

"""Test Reversible Resize"""
resize = Reversible_Resize(128)
resized_img = resize.forward(img)
cv2.imwrite(output_dir + "resized_128.jpg", resized_img)
recover_img = resize.backward(resized_img)
cv2.imwrite(output_dir + "recover_resize_128.jpg", recover_img)

"""Test Reversible Swirl"""
swirl = Reversible_Swirl(center=(H/2, W/2), strength=30, radius=300)
swirled_img = swirl.forward(img)
cv2.imwrite(output_dir + "swirl_100.jpg", swirled_img)
recover_img = swirl.backward(swirled_img)
cv2.imwrite(output_dir + "recover_swirl_256.jpg", recover_img)

"""Test Reversible wavy"""
wavy = Reversible_Wavy(amplitude=15, horizontal=True)
wavy_img = wavy.forward(img)
cv2.imwrite(output_dir + "wavy_horizontal.jpg", wavy_img)
recover_img = wavy.backward(wavy_img)
cv2.imwrite(output_dir + "recover_wavy_horizontal.jpg", recover_img)
