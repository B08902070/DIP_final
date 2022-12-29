import cv2
from pathlib import Path
import numpy as np

from Preserve_Color.Preserve_Color_Matrix import color_transfer


content_img = cv2.imread('images/new-york-city-skyline.jpg', cv2.IMREAD_COLOR)
style_img = cv2.imread('style/3314.jpg', cv2.IMREAD_COLOR)

new_img = color_transfer(content_img=content_img, style_img=style_img)
cv2.imwrite("color_transfer.jpg", new_img)