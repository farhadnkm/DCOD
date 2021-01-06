from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os

root = 'D:/Research data/results/images/rbc/adamw'
img_name = 'amp_adamw_50000.png'
img_out_name = 'amp_adamw_50000_viridis.png'
img = Image.open(os.path.join(root, img_name))
img = img.convert("L")
img = np.array(img)

img = img.astype('float32')
img -= 25
img[img < 0] = 0
img /= np.max(img)
#img /= 255
print(img)
#img /= np.max(img)

cmap = matplotlib.cm.get_cmap('viridis')
img = cmap(img)
img = np.uint8(img * 255)
img = Image.fromarray(img)
img.save(os.path.join(root, img_out_name))
