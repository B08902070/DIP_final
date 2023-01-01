from pathlib import Path
import cv2
import numpy as np

from Preserve_Color.Preserve_Color_Luminance import luminance_only_transfer


content_path = 'content/image_1.jpg'
stylized_path = 'LinearStyleTransfer/Output/image_1_sketch.png'

content_img = cv2.imread(content_path)
stylized_img = cv2.imread(stylized_path)

new_img = luminance_only_transfer(content_img=content_img, stylized_img=stylized_img)
cv2.imwrite("LinearStyleTransfer/Output/image_1_sketch_preserve_color.png", new_img)