from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os

#root = 'D:/Research data/results/images/resolution/plots'
#img_name = 'Artboard 41'
root = 'D:/Research data/results/images/cheek cells'
img_name = '3'
format = '.png'
img = Image.open(os.path.join(root, img_name + format))
img = img.convert("L")
img = np.array(img).astype('float32')
img /= 255
img *= 2
img *= np.pi
img[img<-np.pi] = -np.pi


import skimage.draw as draw

line = draw.line(30, 45, 137, 45)
plt.plot(img[line])
plt.ylim([0.5, 6])
plt.show()
img[line] = 2 * np.pi
plt.imshow(img)
plt.show()
'''
#Artboard 40: [322:368, 246]
#Artboard 41: [261:307, 352]
#Artboard 44: [287:333, 251]
#new: [340:340 + 70, 170:171]
#new2: [350:350 + 46, 204:204 + 1]
#old: [332:332 + 46, 258:258 + 1]

y_start = 260
y_end = 260 + 46
x_start = 352
x_end = 352 + 1
plt.figure(figsize=(4, 2))
plt.plot(img[y_start:y_end, x_start:x_end])
plt.ylim([0, 255])
#plt.yscale('log')
plt.xticks(list(range(0, y_end-y_start, 20)), list(range(y_start, y_end, 20)))
plt.yticks(list(range(0, 255, 125)), list(range(0, 255, 125)))
plt.xlabel('Pixels')
plt.ylabel('Value')
plt.savefig('C:\\Users\\e-min\Desktop\\' + img_name + '_plot' + '.png', transparent=True, bbox_inches='tight', dpi=200)
plt.savefig('C:\\Users\\e-min\Desktop\\' + img_name + '_plot' + '.svg', format='svg', transparent=True, bbox_inches='tight')


img[y_start:y_end, x_start:x_end] = 255
plt.figure()
plt.imshow(img)
plt.show()
#print(np.array(log_adam['loss']))
'''