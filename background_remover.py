import numpy as np
import gdal
import tifffile as tif

hologram_path = 'D:/GoogleDrive_/Colab/Dataset/Custom/cheek/2.tif'
background_path = 'D:/GoogleDrive_/Colab/Dataset/Custom/background1.tif'

hologram_amp = gdal.Open(hologram_path).ReadAsArray().astype('float32')
hologram_amp = hologram_amp[1000:1000+512, 850:850+512]
background = gdal.Open(background_path).ReadAsArray().astype('float32')
background = background[1000:1000+512, 850:850+512]
hologram_amp -= background
hologram_amp -= np.min(hologram_amp)

tif.imsave('C:/Users/e-min\Desktop/results/experiment/universal-256-6 layers/hologram.tiff', hologram_amp.astype('uint16'))