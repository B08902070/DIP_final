from pathlib import Path
import cv2

from Preserve_Color.Preserve_Color_Luminance import luminance_only_transfer


content_img = cv2.imread(str(Path('LinearStyleTransfer/data/content/1.jpg')))
stylized_img = cv2.imread(str(Path('LinearStyleTransfer/Output/1_3314.png')))

new_img = luminance_only_transfer(content_img=content_img, stylized_img=stylized_img)
cv2.imwrite("color_transfer_luminance.jpg", new_img)