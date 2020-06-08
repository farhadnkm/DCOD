import random
import torch
from train import main
from train_utils import Config, HologramDataset, GaussianNoise, RandomCrop, ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if __name__ == '__main__':
	ITERATIONS = 2e5
	# DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	DEVICE = "cpu"
	BATCHSIZE = 2

	AMP_DIR = 'D:\\Research_data\\pap_smear_dataset\\Dataset\\Amplitude'
	PH_DIR = 'D:\\Research_data\\pap_smear_dataset\\Dataset\\Phase'
	AMP_DIR_GT = 'D:\\Research_data\\pap_smear_dataset\\Dataset\\Amplitude_GT'
	PH_DIR_GT = 'D:\\Research_data\\pap_smear_dataset\\Dataset\\Phase_GT'

	LOG_ROOT = 'D:\\Research_data\\train_logs'
	LOG_FOLDER = 'log_02'
	CHECKPOINT_NAME = 'checkpoint.pt.tar'
	LOG_NAME = 'log.csv'
	SUMMARY_NAME = 'summary.txt'

	# random_seed = random.randint(1, 10000)
	random_seed = 999
	print("Random Seed: ", random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)

	transforms_train = transforms.Compose([
		RandomCrop(size=128),
		ToTensor()
	])

	dataset = HologramDataset(amp_dir=AMP_DIR, phase_dir=PH_DIR, amp_dir_gt=AMP_DIR_GT,
							  phase_dir_gt=PH_DIR_GT, transform=transforms_train)
	data_loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4)

	config = Config(dataset=dataset,
					data_loader=data_loader,
					iterations=ITERATIONS,
					batch_size=BATCHSIZE,
					device=DEVICE,
					log_root=LOG_ROOT,
					log_folder=LOG_FOLDER,
					checkpoint_name=CHECKPOINT_NAME)

	main(config)