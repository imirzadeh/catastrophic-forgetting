import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TorchVisionFunc


task_2_perm = np.random.RandomState()
task_3_perm = np.random.RandomState()
task_4_perm = np.random.RandomState()
task_5_perm = np.random.RandomState()
task_states = {2: task_2_perm, 3: task_3_perm, 4: task_4_perm, 5: task_5_perm}

BATCH_SIZE = 64
TOTAL_TASKS = 5

def get_permuted_mnist(task_id, shuffle=False, batch_size=BATCH_SIZE):
	# permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(task_states[task_id].permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) ),
				])
	# torchvision.transforms.Normalize((0.1307,), (0.3081,)) 
	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=shuffle)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=shuffle)

	return train_loader, test_loader


def get_permutaed_mnist_tasks(num_tasks=5, shuffle=False, batch_size=BATCH_SIZE):
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_permuted_mnist(task_id, shuffle, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


class RotationTransform:
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, shuffle=False, batch_size=BATCH_SIZE):
	ROTATE_DEGREES_PER_TASK = 10
	rotation_angle = (task_id-1)*ROTATE_DEGREES_PER_TASK

	transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_angle),
		torchvision.transforms.ToTensor(),
		])

	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=shuffle)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=shuffle)

	return train_loader, test_loader


def get_rotated_mnist_tasks(num_tasks=5, shuffle=False, batch_size=BATCH_SIZE):
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_rotated_mnist(task_id, shuffle, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets
