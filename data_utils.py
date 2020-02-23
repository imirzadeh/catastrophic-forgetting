import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader


task_2_perm = np.random.RandomState(2222)
task_3_perm = np.random.RandomState(3333)
task_4_perm = np.random.RandomState(4444)
task_5_perm = np.random.RandomState(5555)
task_states = {2: task_2_perm, 3: task_3_perm, 4: task_4_perm, 5: task_5_perm}

BATCH_SIZE = 128
TOTAL_TASKS = 5

def get_permuted_mnist(task_id):
	# permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(task_states[task_id].permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) ),
				torchvision.transforms.Normalize((0.1307,), (0.3081,)) ])

	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=BATCH_SIZE, shuffle=False)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=BATCH_SIZE, shuffle=False)

	return train_loader, test_loader


def get_mnist_tasks():
	datasets = {}
	for task_id in range(1, 6):
		train_loader, test_loader = get_permuted_mnist(task_id)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}

	print(datasets.keys())
	return datasets