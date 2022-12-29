from pathlib import Path
import cv2
import numpy as np

from Preserve_Color.Preserve_Color_Luminance import luminance_only_transfer


content_path = 'images/image_1.jpg'
stylized_path = 'MCCNet/out/image_1/image_1_rock.jpg'

content_img = cv2.imread(content_path)
stylized_img = cv2.imread(stylized_path)

new_img = luminance_only_transfer(content_img=content_img, stylized_img=stylized_img)
cv2.imwrite("color_preserve_image_1_rock.jpg", new_img)