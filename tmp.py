import imageio
from pathlib import Path
img = imageio.imread(Path('images/image_0.jpg'))
print(type(img))