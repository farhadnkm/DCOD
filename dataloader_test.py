from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from utils.data import HologramDataset
from utils.transforms import RandomCrop, ToTensor, GaussianNoise


def show_batch(dataloader, batch_index, sample_index):
	fig, axes = plt.subplots(1, 4)

	for i, sample_batched in enumerate(dataloader):
		if i == batch_index:
			print(i, sample_batched['input'].size(),
				  sample_batched['output'].size())
			axes[0].imshow(sample_batched['input'][sample_index, 0, :, :], cmap='gray')
			axes[1].imshow(sample_batched['input'][sample_index, 1, :, :], cmap='gray')
			axes[2].imshow(sample_batched['output'][sample_index, 0, :, :], cmap='gray')
			axes[3].imshow(sample_batched['output'][sample_index, 1, :, :], cmap='gray')
			break

	plt.show()

def show_batch_single_channel(dataloader, batch_index, sample_index):
	fig, axes = plt.subplots(1, 2)

	for i, sample_batched in enumerate(dataloader):
		if i == batch_index:
			print(i, sample_batched['input'].size(),
				  sample_batched['output'].size())
			axes[0].imshow(sample_batched['input'][sample_index, 0, :, :], cmap='gray')
			axes[1].imshow(sample_batched['output'][sample_index, 0, :, :], cmap='gray')
			break

	plt.show()

if __name__ == '__main__':  # <== this line is necessary

	AMP_DIR = 'D:\Research_data\pap_smear_dataset\Dataset\Amplitude'
	PH_DIR = 'D:\Research_data\pap_smear_dataset\Dataset\Phase'
	AMP_DIR_GT = 'D:\Research_data\pap_smear_dataset\Dataset\Amplitude_GT'
	PH_DIR_GT = 'D:\Research_data\pap_smear_dataset\Dataset\Phase_GT'

	transforms_train = transforms.Compose([
		RandomCrop(size=128),
		ToTensor()
	])
	dataset = HologramDataset(amp_dir=AMP_DIR, phase_dir=PH_DIR, amp_dir_gt=AMP_DIR_GT,
							  phase_dir_gt=PH_DIR_GT, transform=transforms_train)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
	#show_batch(dataloader, batch_index=0, sample_index=0)
	show_batch_single_channel(dataloader, batch_index=0, sample_index=0)