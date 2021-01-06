from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from utils.process import Simulator
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def GTImages(tag="default", *args):
	if tag == "default":
		amp_path = 'D:/Research data/results/images/selected/Simulation/default/out/out_amp.png'
		ph_path = 'D:/Research data/results/images/selected/Simulation/default/out/out_phase.png'

	if tag == "contrast_experiment":
		for value in args:
			if value == "100":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 2/100/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 2/100/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 2/100.png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "75":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 2/75/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 2/75/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 2/75.png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "50":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 2/50/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 2/50/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 2/50.png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "25":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 2/25/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 2/25/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 2/25.png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "0":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 2/0/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 2/0/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 2/0.png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'

	if tag == "blur_experiment":
		for value in args:
			if value == "0":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 1/0/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 1/0/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 1/0.png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "exp(0)":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(0)/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(0)/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(0).png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "exp(1)":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(1)/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(1)/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(1).png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "exp(2)":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(2)/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(2)/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(2).png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'
			if value == "exp(3)":
				amp_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(3)/out_amp.png'
				ph_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(3)/out_phase.png'
				gt_apm_path = 'D:/Research data/results/images/selected/Simulation/test 1/exp(3).png'
				gt_ph_path = 'D:/Research data/results/images/selected/Simulation/phase.png'

	img_amp = Image.open(amp_path)
	img_amp = img_amp.convert("L")
	img_amp = np.array(img_amp).astype("float32")
	img_amp /= 255

	img_ph = Image.open(ph_path)
	img_ph = img_ph.convert("L")
	img_ph = np.array(img_ph).astype("float32")
	img_ph /= 255

	gt_img_amp = Image.open(gt_apm_path)
	gt_img_amp = gt_img_amp.convert("L")
	gt_img_amp = np.array(gt_img_amp).astype("float32")
	gt_img_amp /= 255

	gt_img_ph = Image.open(gt_ph_path)
	gt_img_ph = gt_img_ph.convert("L")
	gt_img_ph = np.array(gt_img_ph).astype("float32")
	gt_img_ph /= 255

	return img_amp, img_ph, gt_img_amp, gt_img_ph


def Scale(img, perc, max_val):
	img = img.copy()
	img *= perc
	img += 1 - perc
	img /= max_val
	return img


def Scale_back(img, perc, max_val):
	img = img.copy()
	img -= 1 - perc
	img[img < 0] = 0
	img *= 1 / perc
	img /= max_val
	return img


def NRMS(x, ref):
	return np.sqrt(np.mean(x - ref)) / np.sqrt(np.mean(ref))


def SSIM(img, img_gt):
	return ssim(img, img_gt, data_range=img_gt.max() - img_gt.min())


def NSTD(img):
	mean = np.mean(img)
	min_gt = np.min(img)
	max_gt = np.max(img)
	return np.sqrt(np.mean(np.square((img - mean)/(max_gt - min_gt))))


def PSE(img):
	x, y = np.shape(img)[0], np.shape(img)[1]
	X = np.abs(fft2(img))
	psd = np.square(X)/(x * y)
	npsd = psd / sum(psd)
	return -np.sum(npsd * np.log2(npsd))


#img_amp, img_ph, gt_img_amp, gt_img_ph = GTImages("contrast_experiment", "0")
img_amp, img_ph, gt_img_amp, gt_img_ph = GTImages("blur_experiment", "exp(3)")

#sim = Simulator(np.shape(img_amp), 1.12, 1.12, 5.32e-3)
#rec = sim.reconstruct(gt_img_amp * np.exp(1j * gt_img_ph))
#rec_amp = np.abs(rec)

#print("NRMS:", NRMS(rec_amp**2, np.ones_like(rec_amp))
#print("PSE:", PSE(gt_img_amp))
print("amplitude:", SSIM(img_amp, gt_img_amp))
print("phase:", SSIM(img_ph, gt_img_ph))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(img_amp, cmap='gray', vmin=0, vmax=1)
ax2.imshow(img_ph, cmap='viridis', vmin=0, vmax=1)
ax3.imshow(gt_img_amp, cmap='gray', vmin=0, vmax=1)
ax4.imshow(gt_img_ph, cmap='viridis', vmin=0, vmax=1)
plt.show()

""" RBC SSIM compared with MHPR
30000 iters:
	amp: 0.5498982938470156
	ph:  0.8294845548999826
35000 iters:
	amp: 0.5255979257550196
	ph:  0.8279558746422462
"""

""" RBC holograms
Adam:
	PSNR: 32.022479272961405
	SSIM: 0.9999697240715613
AdamW:
	PSNR: 28.440033670170997
	SSIM: 0.9999318830633466
AdamWR:
	PSNR: 31.886596019184875
	SSIM: 0.9999438316377467
"""