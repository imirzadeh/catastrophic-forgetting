from comet_ml import Experiment

import os
import argparse
import nni
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from models import ResNet18
from data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks,get_split_cifar100_tasks_2
matplotlib.style.use('ggplot')
	
DEVICE = 'cuda'

def parse_arguments():
	parser = argparse.ArgumentParser(description='Arg parser')
	parser.add_argument('--hidden_size', default=256, type=int, help='num hiddens')
	args = parser.parse_args()
	return args



def train_single_epoch(net, optimizer, loader, task_id, config):
	net = net.to(DEVICE)
	net.train()
	
	criterion = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		pred = net(data, task_id)#net(data, task_id)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net


def eval_single_epoch(net, loader, task_id):
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	crit = nn.CrossEntropyLoss()
	cnt = 0
	with torch.no_grad():
		for data, target in loader:
			cnt += 1

			data = data.to(DEVICE)
			target = target.to(DEVICE)

			output = net(data, task_id)#net(data, task_id)
			# if cnt == 1:
			#	print(output.data.max(1, keepdim=True)[1][:20])
			test_loss += crit(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(loader.dataset),
			100. * correct / len(loader.dataset)))

	result = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return result

def record_firing_rate(net, loader):
	net = net.to(DEVICE)
	net.eval()
	fires_l1, fires_l2 = None, None
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			if fires_l1 is None:
				fires_l1, fires_l2 = net.get_firing_acts(data)
			else:
				l1, l2 = net.get_firing_acts(data)
				fires_l1 += l1
				fires_l2 += l2
	return fires_l1.numpy(), fires_l2.numpy()


def visualize_result(df, filename):
	ax = sns.lineplot(data=df,  dashes=False)
	ax.figure.savefig('./stash/{}.png'.format(filename), dpi=350)


def save_net(net, filename):
	torch.save(net.state_dict(), filename)

def save_firing_patterns(patterns, filename):
	df = pd.DataFrame(patterns)
	df.to_csv(filename)

if __name__ == "__main__":
	trial_id = os.environ.get('NNI_TRIAL_JOB_ID', "UNKNOWN")
	args = parse_arguments()
	experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", auto_param_logging=False, auto_metric_logging=False, 
						project_name="explore-cifar", workspace="nn-forget", disabled=False)

	hidden_size = args.hidden_size
	config = nni.get_next_parameter()

	# config = {'epochs': 1, 'dropout_1': 0.2, 'dropout_2':0.2, 'lr': 0.1, 'gamma': 0.1, 'lr_lb': 0.005}
	config['trial'] = trial_id
	config['hidden_size'] = hidden_size
	# lr = max(config['lr']*(config['gamma']**task_id), config['lr_lb'])#0.015* 0.6**(task_id)
	TASKS = 5


	#net = MLP(hidden_layers=[hidden_size, hidden_size, 10], config=config).to(DEVICE)
	net = ResNet18().to(DEVICE)
	tasks = get_split_cifar100_tasks_2(TASKS)
	optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.8)
	scheduler = MultiStepLR(optimizer, milestones=[1, 2, 3, 4], gamma=config['gamma'])
	template = {i: [] for i in range(1, TASKS+1)}
	running_test_accs = copy.deepcopy(template)
	

	firing_patterns_l1 = copy.deepcopy(template)
	firing_patterns_l2 = copy.deepcopy(template)
	firing_patterns_l1_1 = copy.deepcopy(template)
	firing_patterns_l2_1 = copy.deepcopy(template)
	
	experiment.log_parameters(config)

	for task_id in range(1, TASKS+1):
		print("======================= TASK {} =======================".format(task_id))
		task_data = tasks[task_id]
		train_loader, test_loader = task_data['train'], task_data['test']
		prev_train_loader = None
		if task_id >= 2:
			task_data_prev = tasks[task_id-1]
			prev_train_loader, prev_test_loader = task_data_prev['train'], task_data_prev['test']
		for epoch in range(1, config['epochs']+1):
			print(">>> epoch {}".format(epoch))
			# train
			net = train_single_epoch(net, optimizer, train_loader, task_id, config)
			# eval
			for test_task_id in range(1, TASKS+1):#[1, 2, 3]:#range(1, TASKS+1):
				if test_task_id > task_id:
					test_acc = 0 # left-padding with zero
				else:
					test_acc = eval_single_epoch(net, tasks[test_task_id]['test'], test_task_id)
				running_test_accs[test_task_id].append(test_acc)
		scheduler.step(task_id)

	score = np.mean([running_test_accs[i][-1] for i in running_test_accs.keys()])
	# score = (running_test_accs[1][-1] + running_test_accs[1][-2] + running_test_accs[1][-3])/3.0
	forget = np.mean([max(running_test_accs[i])-running_test_accs[i][-1] for i in range(1, TASKS)])/100.0
	experiment.log_metric(name='score', value=score)
	experiment.log_metric(name='forget', value=forget)

	clone = []
	for k in running_test_accs.keys():
		dic = {'name': str(k), 'data': running_test_accs[k]}
		clone.append(dic)

	df = pd.DataFrame(running_test_accs)
	df.to_csv('./stash/{}.csv'.format(trial_id))
	print(score)
	nni.report_final_result(score)
	experiment.log_asset('./stash/{}.csv'.format(trial_id))
	visualize_result(df, trial_id)
	experiment.log_asset('./stash/{}.png'.format(trial_id))
	experiment.log_figure('./stash/{}.png'.format(trial_id))
	
	experiment.end()

