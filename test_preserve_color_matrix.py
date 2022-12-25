import imageio
from pathlib import Path
import numpy as np

from Preserve_Color.Preserve_Color_Matrix import color_transfer


content_img = imageio.imread(Path('LinearStyleTransfer/data/content/1.jpg'))
style_img = imageio.imread(Path('LinearStyleTransfer/data/style/27.jpg'))

new_img = color_transfer(content_img=content_img, style_img=style_img)
imageio.imwrite("color_transfer.jpg", new_img)