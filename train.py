from comet_ml import Experiment
import os
import argparse
import nni
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_utils import get_mnist_tasks
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

DEVICE = 'cpu'

def parse_arguments():
	parser = argparse.ArgumentParser(description='Arg parser')
	# parser.add_argument('--epochs', default=21000, type=int,  help='number of total epochs to run')
	parser.add_argument('--hidden_size', type=int, help='num hiddens')
	args = parser.parse_args()
	return args

class MLP(nn.Module):
	def __init__(self, hidden_layers):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(784, hidden_layers[0])
		self.relu = nn.LeakyReLU(0.01)
		self.W2 = nn.Linear(hidden_layers[0], hidden_layers[1])
		self.W3 = nn.Linear(hidden_layers[1], hidden_layers[2])
	
	def forward(self, x):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.W2(out)
		out = self.relu(out)
		out = self.W3(out)
		return out
	
	def get_layer1_acts(self, x, pre_act=False):
		x = x.view(-1, 784)
		out = self.W1(x)
		if not pre_act:
			out = self.relu(out)
		out = out.detach().numpy().T
		return out
	
	def get_layer2_acts(self, x):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.W2(out)
		# out = self.relu(out)
		out = out.detach().numpy().T
		return out


def train_single_epoch(net, loader):
	net = net.to(DEVICE)
	net.train()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
	criterion = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		pred = net(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net


def train_WAS_single_epoch(net, current_task_loader, prev_task_loader=None):
	net = net.to(DEVICE)
	net.train()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
	criterion = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(current_task_loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		pred = net(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()

	if prev_task_loader:
		for batch_idx, (data, target) in enumerate(current_task_loader):
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			optimizer.zero_grad()
			pred = net(data)
			loss = criterion(pred, target)
			loss.backward()
			optimizer.step()
	return net


def eval_single_epoch(net, loader):
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	crit = nn.CrossEntropyLoss()
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			output = net(data)
			test_loss += crit(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	#experiment.log_metric(name='val-acc', step=epoch, value=(float(correct.numpy())*100.0)/10000.0)
	#experiment.log_metric(name='val-loss', step=epoch, value=test_loss)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(loader.dataset),
			100. * correct / len(loader.dataset)))

	result = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return result

def visualize_result(df, filename):
	ax = sns.lineplot(data=df)
	ax.figure.savefig('./stash/{}.png'.format(filename), dpi=350)


if __name__ == "__main__":
	trial_id = os.environ.get('NNI_TRIAL_JOB_ID', "NULL")
	args = parse_arguments()
	experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", auto_param_logging=False, auto_metric_logging=False, 
						project_name="explore", workspace="nn-forget")

	hidden_size = args.hidden_size
	config = nni.get_next_parameter()
	config['trial'] = trial_id
	config['hidden_size'] = hidden_size
	



	net = MLP(hidden_layers=[hidden_size, hidden_size, 10]).to(DEVICE)
	tasks = get_mnist_tasks()

	running_test_accs = {1: [], 2:[], 3:[], 4:[], 5:[]}

	for task_id in range(1, 6):
		print("======================= TASK {} =======================".format(task_id))
		task_data = tasks[task_id]
		train_loader, test_loader = task_data['train'], task_data['test']
		prev_train_loader = None
		if task_id >= 2:
			task_data_prev = tasks[task_id-1]
			prev_train_loader, prev_test_loader = task_data_prev['train'], task_data_prev['test']
		for epoch in range(1, 11):
			print(">>> epoch {}".format(epoch))
			# train
			if config['train_policy'] == 1:
				net = train_single_epoch(net, train_loader)
			else:
				net = train_WAS_single_epoch(net, train_loader, prev_train_loader)

			# eval
			for test_task_id in range(1, 6):
				if test_task_id > task_id:
					test_acc = 0 # left-padding with zero
				else:
					test_acc = eval_single_epoch(net, tasks[test_task_id]['test'])
				running_test_accs[test_task_id].append(test_acc)


	score = (running_test_accs[1][-1] + running_test_accs[1][-2] + running_test_accs[1][-3])/3.0

	# log everything
	experiment.log_parameters(config)
	experiment.log_metric(name='score', value=score)

	df = pd.DataFrame(running_test_accs)
	df.to_csv('./stash/{}.csv'.format(trial_id))
	visualize_result(df, trial_id)
	nni.report_final_result(score)
	experiment.log_asset('./stash/{}.csv'.format(trial_id))
	experiment.log_asset('./stash/{}.png'.format(trial_id))
	experiment.log_figure('./stash/{}.png'.format(trial_id))
	experiment.end()

