from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from fringe.utils.io import import_image
from fringe.utils.modifiers import ImageToArray
from fringe.process.cpu import AngularSpectrumSolver as AsSolver


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

	p1 = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
	img_amp = import_image(path=amp_path, preprocessor=p1)
	img_ph = import_image(path=ph_path, preprocessor=p1)
	gt_img_amp = import_image(path=gt_apm_path, preprocessor=p1)
	gt_img_ph = import_image(path=gt_ph_path, preprocessor=p1)

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

'''
#img_amp, img_ph, gt_img_amp, gt_img_ph = GTImages("contrast_experiment", "0")
img_amp, img_ph, gt_img_amp, gt_img_ph = GTImages("blur_experiment", "exp(3)")

solver = AsSolver(shape=np.shape(img_amp), dx=1.12, dy=1.12, wavelength=5.32e-3)
h = gt_img_amp * np.exp(1j * gt_img_ph)
rec = solver.solve(h, 300)
rec_amp = np.abs(rec)

print("NRMS:", NRMS(rec_amp**2, np.ones_like(rec_amp)))
print("PSE:", PSE(gt_img_amp))
print("amplitude:", SSIM(img_amp, gt_img_amp))
print("phase:", SSIM(img_ph, gt_img_ph))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(img_amp, cmap='gray', vmin=0, vmax=1)
ax2.imshow(img_ph, cmap='viridis', vmin=0, vmax=1)
ax3.imshow(gt_img_amp, cmap='gray', vmin=0, vmax=1)
ax4.imshow(gt_img_ph, cmap='viridis', vmin=0, vmax=1)
plt.show()'''