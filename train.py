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
from data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

DEVICE = 'cuda'

def parse_arguments():
	parser = argparse.ArgumentParser(description='Arg parser')
	parser.add_argument('--hidden_size', default=100, type=int, help='num hiddens')
	args = parser.parse_args()
	return args

class MLP(nn.Module):
	def __init__(self, hidden_layers, config):
		super(MLP, self).__init__()
		self.num_time_tensors = 50
		# self.W1 = nn.Linear(784+self.num_time_tensors, hidden_layers[0])
		self.W1 = nn.Linear(784, hidden_layers[0])
		# self.relu = nn.LeakyReLU(0.01)	
		self.relu = nn.ReLU()
		self.W2 = nn.Linear(hidden_layers[0], hidden_layers[1])
		self.W3 = nn.Linear(hidden_layers[1], hidden_layers[2])
		self.dropout_1 = nn.Dropout(p=config['dropout_1'])
		self.dropout_2 = nn.Dropout(p=config['dropout_2'])
		self.batchnorm = nn.BatchNorm1d(hidden_layers[0])
	
	def get_firing_acts(self, x):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		l1 = torch.sum((out > 0.0).float(), dim=0)
		out = self.W2(out)
		out = self.relu(out)
		l2 = torch.sum((out > 0.0).float(), dim=0)
		return l1.cpu(), l2.cpu()

	def forward(self, x, task_id):
		# ratio = 0.05
		x = x.view(-1, 784)
		# x = torch.cat((ratio*task_id*torch.ones((x.shape[0], self.num_time_tensors)), x), dim=1)
		out = self.W1(x)
		out = self.relu(out)
		out = self.dropout_1(out)
		# out = torch.cat((ratio*task_id*torch.ones((out.shape[0], self.num_time_tensors)), out), dim=1)
		out = self.W2(out)
		out = self.relu(out)
		out = self.dropout_2(out)
		out = self.W3(out)
		return out


def train_single_epoch(net, loader, task_id, config):
	net = net.to(DEVICE)
	net.train()
	lr = max(config['lr']*(config['gamma']**task_id), config['lr_lb'])#0.015* 0.6**(task_id)
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)
	criterion = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		pred = net(data, task_id)
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
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			output = net(data, task_id)
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
						project_name="permuted-long", workspace="nn-forget", disabled=False)

	hidden_size = args.hidden_size
	config = nni.get_next_parameter()

	# config = {'epochs': 5, 'dropout_1': 0.2, 'dropout_2':0.2, 'lr': 0.002, 'gamma': 0.9, 'lr_lb': 0.005}
	config['trial'] = trial_id
	config['hidden_size'] = hidden_size
	
	TASKS = 20

	net = MLP(hidden_layers=[hidden_size, hidden_size, 10], config=config).to(DEVICE)
	tasks = get_permuted_mnist_tasks(TASKS)

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
			net = train_single_epoch(net, train_loader, task_id, config)
			# eval
			for test_task_id in range(1, TASKS+1):#[1, 2, 3]:#range(1, TASKS+1):
				if test_task_id > task_id:
					test_acc = 0 # left-padding with zero
				else:
					test_acc = eval_single_epoch(net, tasks[test_task_id]['test'], test_task_id)
				running_test_accs[test_task_id].append(test_acc)
		firing_patterns_l1_1[task_id], firing_patterns_l2_1[task_id] = record_firing_rate(net, tasks[1]['test'])
		firing_patterns_l1[task_id], firing_patterns_l2[task_id] = record_firing_rate(net, tasks[task_id]['test'])
	
	save_firing_patterns(firing_patterns_l1_1, './stash/id={}-firing-history-l1-1.csv'.format(trial_id))
	save_firing_patterns(firing_patterns_l2_1, './stash/id={}-firing-history-l2-1.csv'.format(trial_id))
	save_firing_patterns(firing_patterns_l1, './stash/id={}-firing-history-l1.csv'.format(trial_id))
	save_firing_patterns(firing_patterns_l2, './stash/id={}-firing-history-l2.csv'.format(trial_id))

	score = np.mean([running_test_accs[i][-1] for i in running_test_accs.keys()])
	# score = (running_test_accs[1][-1] + running_test_accs[1][-2] + running_test_accs[1][-3])/3.0
	
	experiment.log_metric(name='score', value=score)

	clone = []
	for k in running_test_accs.keys():
		dic = {'name': str(k), 'data': running_test_accs[k]}
		clone.append(dic)

	df = pd.DataFrame(running_test_accs)
	df.to_csv('./stash/{}.csv'.format(trial_id))
	print(score)
	nni.report_final_result(score)
	experiment.log_asset('./stash/{}.csv'.format(trial_id))
	experiment.log_asset('./stash/id={}-firing-history-l1.csv'.format(trial_id))
	experiment.log_asset('./stash/id={}-firing-history-l2.csv'.format(trial_id))
	experiment.log_asset('./stash/id={}-firing-history-l1-1.csv'.format(trial_id))
	experiment.log_asset('./stash/id={}-firing-history-l2-1.csv'.format(trial_id))
	visualize_result(df, trial_id)
	experiment.log_asset('./stash/{}.png'.format(trial_id))
	experiment.log_figure('./stash/{}.png'.format(trial_id))
	
	experiment.end()

