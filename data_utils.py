import numpy as np
import torch
import torchvision
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TorchVisionFunc


task_2_perm = np.random.RandomState()
task_3_perm = np.random.RandomState()
task_4_perm = np.random.RandomState()
task_5_perm = np.random.RandomState()
# task_states = {2: task_2_perm, 3: task_3_perm, 4: task_4_perm, 5: task_5_perm,
# 			 6: task_2_perm, 7: task_3_perm, 8: task_4_perm, 9: task_5_perm}

task_states = {i: np.random.RandomState() for i in range(2, 10)}
BATCH_SIZE = 64

def get_permuted_mnist(task_id, shuffle=False, batch_size=BATCH_SIZE):
	# permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(np.random.RandomState().permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) ),
				])
	# torchvision.transforms.Normalize((0.1307,), (0.3081,)) 
	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

	return train_loader, test_loader


def get_permuted_mnist_tasks(num_tasks=5, shuffle=False, batch_size=BATCH_SIZE):
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



def get_split_cifar100(task_id, batch_size=BATCH_SIZE, shuffle=False):
	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	start_class = (task_id-1)*5
	end_class = task_id * 5

	transforms = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=transforms)
	test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=transforms)
	
	targets_train = torch.tensor(train.targets)
	target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

	targets_test = torch.tensor(test.targets)
	target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

	train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(train, np.where(target_train_idx==1)[0]), batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx==1)[0]), batch_size=batch_size)

	return train_loader, test_loader


def get_split_cifar100_tasks(num_tasks, shuffle=False, batch_size=BATCH_SIZE):
	datasets = {}
	for task_id in range(1, num_tasks):
		train_loader, test_loader = get_split_cifar100(task_id, batch_size, shuffle)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


if __name__ == "__main__":
	dataset = get_split_cifar100_tasks()




