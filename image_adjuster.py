from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os

path = 'D:\GoogleDrive_\Colab\Logs\log_35_test_25p_ddn'

img = np.array(Image.open(os.path.join(path, 'out_amp.png'))).astype('float32')
img /= np.max(img)
img *= 1.1

scale = 0.25
img -= (1-scale)
img *= 1/(scale)
img[img < 0] = 0
img[img > 1] = 1

img /= 1.1

plt.imshow(img, vmin=0, vmax=1, cmap='gray')
plt.show()

img = np.uint8(img * 255)
img = Image.fromarray(img)
img = img.convert('L')
img.save(os.path.join(path, 'out_amp_scaled.png'))