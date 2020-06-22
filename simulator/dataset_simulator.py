import numpy as np
import PIL.Image as Image
import tifffile as tiff
import cv2
from skimage.restoration import unwrap_phase
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import simulator.simulation_utils as utils


def getkey(fname):
	return int(fname.split('_')[0])


def sorted_alphanumeric(flist):
	return sorted(flist, key=getkey)


PATH_AMP = 'D:\Research_data\pap_smear_dataset\Intensity'
PATH_PH = 'D:\Research_data\pap_smear_dataset\Phase'
OUT_PATH_AMP = 'D:\Research_data\pap_smear_dataset\Dataset\Amplitude'
OUT_PATH_PH = 'D:\Research_data\pap_smear_dataset\Dataset\Phase'
OUT_PATH_AMP_GT = 'D:\Research_data\pap_smear_dataset\Dataset\Amplitude_GT'
OUT_PATH_PH_GT = 'D:\Research_data\pap_smear_dataset\Dataset\Phase_GT'


amps = sorted_alphanumeric(os.listdir(PATH_AMP))
phs = sorted_alphanumeric(os.listdir(PATH_PH))

scale = 0.25

object_funcs = []
print('Recording ground truth images by index')
for i in range(len(amps)):
	amp = np.asarray(Image.open(os.path.join(PATH_AMP, amps[i]))).astype('float32')
	ph = tiff.imread(os.path.join(PATH_PH, phs[i])).astype('float32') / 2200
	x, y = round(amp.shape[0] * scale), round(amp.shape[1] * scale)

	amp = cv2.resize(amp, dsize=(x, y))
	ph = cv2.resize(ph, dsize=(x, y))
	object_funcs.append(amp * np.exp(1j * ph))

	amp_gt = Image.fromarray(amp).convert('L')
	amp_gt.save(os.path.join(OUT_PATH_AMP_GT, str(i)+'.png'))
	ph_gt = ph * 2200 * 255 / 65536
	ph_gt = Image.fromarray(ph_gt).convert('L')
	ph_gt.save(os.path.join(OUT_PATH_PH_GT, str(i) + '.png'))

	print(i)



delta_x = 0.5 / scale
delta_y = 0.5 / scale
W = 532.2e-3
shape = object_funcs[0].shape
simulator = utils.Simulator(shape=shape, delta_x=delta_x, delta_y=delta_y, w=W)

np.random.seed(999)
zs = np.random.normal(150, 50, len(object_funcs))

print('Recording holographic images')
for i in range(len(object_funcs)):
	z = zs[i]
	res = simulator.reconstruct(initializer=object_funcs[i], z=z)
	res_abs = np.abs(res)
	res2 = simulator.reconstruct(initializer=res_abs, z=-z)

	amp = np.abs(res2)
	amp /= np.max(amp)
	amp *= 255
	amp = Image.fromarray(amp).convert('L')
	amp.save(os.path.join(OUT_PATH_AMP, str(i)+'.png'))

	ph = np.angle(res2)
	ph = unwrap_phase(ph)
	ph -= np.mean(ph)
	ph += np.pi
	ph /= 2 * np.pi
	ph *= 255
	ph = Image.fromarray(ph).convert('L')
	ph.save(os.path.join(OUT_PATH_PH, str(i) + '.png'))

	print(i)
