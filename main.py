import os
import nni
import uuid
import numpy as np
import pandas as pd
from comet_ml import Experiment
import torch
import torch.nn as nn
from models import MLP
from hessian_eigenthings import compute_hessian_eigenthings
from pathlib import Path
from utils import visualize_result
from models import ResNet18
from data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks, get_split_cifar100_tasks


# config = nni.get_next_parameter()
# config = {'epochs': 5, 'dropout': 0.25,
# 		 'batch_size': 64, 'lr': 0.1, 'gamma': 0.5,
# 		 'lrlb': 0.00001, 'momentum': 0.8}

# dataset = 'perm' if np.random.randint(2) == 1 else 'rot'
first_stable = True if np.random.randint(2) == 1 else False
second_stable = True if np.random.randint(2) == 1 else False

config_stable   = {'lr': 0.15,  'gamma': 0.3, 'momentum': 0.8, 'lrlb': 0.0001, 'dropout': 0.25, 'batch_size': 32, 'epochs': 5}
config_unstable = {'lr': 0.05, 'gamma': 1.0, 'momentum': 0.8, 'lrlb': 0.0001, 'dropout': 0.0, 'batch_size': 256, 'epochs': 5}



TRIAL_ID = os.environ.get('NNI_TRIAL_JOB_ID', uuid.uuid4().hex.upper()[0:6])
# config['trial'] = TRIAL_ID
EXPERIMENT_DIRECTORY = './outputs/{}'.format(TRIAL_ID)
DEVICE = 'cuda'				

# =============== SETTINGS ================
NUM_TASKS = 2
NUM_EIGENS = 1
EPOCHS = config_stable['epochs']
# HIDDENS = config['hiddens']
HIDDENS = 100
# BATCH_SIZE = config['batch_size']
experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg",
						project_name="nips-lambda2-perm-3",
						auto_param_logging=False, auto_metric_logging=False,
						workspace="nn-forget", disabled=False)

loss_db = {t:[0 for i in range(NUM_TASKS*EPOCHS)] for t in range(1, NUM_TASKS+1)}
acc_db = {t:[0 for i in range(NUM_TASKS*EPOCHS)] for t in range(1, NUM_TASKS+1)}
hessian_eig_db = {}


def setup_experiment():
	print('Experiment started')
	# experiment.log_parameters(config)
	experiment.log_parameters({'first_stable': first_stable, 'second_stable': second_stable})
	Path(EXPERIMENT_DIRECTORY).mkdir(parents=True, exist_ok=True)

def end_experiment():
	acc_df = pd.DataFrame(acc_db)
	acc_df.to_csv(EXPERIMENT_DIRECTORY+'/accs.csv')
	visualize_result(acc_df, EXPERIMENT_DIRECTORY+'/accs.png')

	loss_df = pd.DataFrame(loss_db)
	loss_df.to_csv(EXPERIMENT_DIRECTORY+'/loss.csv')
	visualize_result(loss_df, EXPERIMENT_DIRECTORY+'/loss.png')

	hessian_df = pd.DataFrame(hessian_eig_db)
	hessian_df.to_csv(EXPERIMENT_DIRECTORY+'/hessian_eigs.csv')

	score = np.mean([acc_db[i][-1] for i in acc_db.keys()])
	forget = np.mean([max(acc_db[i])-acc_db[i][-1] for i in range(1, NUM_TASKS)])/100.0

	print('score = {}, forget = {}'.format(score, forget))
	experiment.log_metric(name='score', value=score)
	experiment.log_metric(name='forget', value=forget)
	experiment.log_asset_folder(EXPERIMENT_DIRECTORY)
	experiment.end()

def log_metrics(metrics, time, task_id):
	print('epoch {}, metrics: {}'.format(time, metrics))
	# log to db
	acc = metrics['accuracy']
	loss = metrics['loss']
	loss_db[task_id][time-1] = loss
	acc_db[task_id][time-1] = acc

	# log to comet
	experiment.log_metric(name='task {} - loss'.format(task_id), step=time-1, value=loss)
	experiment.log_metric(name='task {} - acc'.format(task_id), step=time-1, value=acc)

def save_eigenvec(filename, arr):
	np.save(filename, arr)

def log_hessian(model, loader, time, task_id):
	criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
	use_gpu = True if DEVICE != 'cpu' else False
	est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
		model,
		loader,
		criterion,
		num_eigenthings=NUM_EIGENS,
		power_iter_steps=18,
		power_iter_err_threshold=1e-5,
		momentum=0,
		use_gpu=True,
	)
	key = 'task-{}-epoch-{}'.format(task_id, time-1)
	hessian_eig_db[key] = est_eigenvals
	save_eigenvec(EXPERIMENT_DIRECTORY+"/{}-vec.npy".format(key), est_eigenvecs)
	experiment.log_histogram_3d(name='task-{}-eigs'.format(task_id), step=time-1, values=est_eigenvals)

def save_checkpoint(model, time):
	filename = '{directory}/model-{trial}-{time}.pth'.format(directory=EXPERIMENT_DIRECTORY, trial=TRIAL_ID, time=time)
	torch.save(model.cpu().state_dict(), filename)

def train_single_epoch(net, optimizer, loader, criterion, task_id=None):
	net = net.to(DEVICE)
	net.train()
	
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred = net(data, task_id)
		else:
			pred = net(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net

def eval_single_epoch(net, loader, criterion, task_id=None):
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			# for cifar head
			if task_id:
				output = net(data, task_id)
			else:
				output = net(data)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}

def run():
	configs = []
	if first_stable:
		configs.append(config_stable)
	else:
		configs.append(config_unstable)

	if second_stable:
		configs.append(config_stable)
	else:
		configs.append(config_unstable)

	# # basics
	# model = MLP([HIDDENS, HIDDENS, 10], config=config).to(DEVICE)
	# # model = ResNet18(100, 20, config=config).to(DEVICE)
	# tasks = get_rotated_mnist_tasks(NUM_TASKS, shuffle=True, batch_size=BATCH_SIZE)
	
	# optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['gamma'])
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	model = MLP([HIDDENS, HIDDENS, 10], config=configs[0]).to(DEVICE)

	# hooks
	setup_experiment()
	# main loop
	time = 0
	for current_task_id in range(1, NUM_TASKS+1):
		# basics
		config = configs[current_task_id-1]
		model.dropout_p = config['dropout']


		tasks = get_permuted_mnist_tasks(NUM_TASKS, shuffle=True, batch_size=config['batch_size'])
		print("========== TASK {} / {} ============".format(current_task_id, NUM_TASKS))
		train_loader =  tasks[current_task_id]['train']
		# task_lr = config['lr']*(config['gamma']**(current_task_id-1))
		for epoch in range(1, EPOCHS+1):
			# train and save
			lr = max(config['lr']* config['gamma']**(epoch), config['lrlb'])
			print(lr)

			optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config['momentum'])
			train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1

			# evaluate on all tasks up to now
			for prev_task_id in range(1, current_task_id+1):
				if epoch == EPOCHS:
					model = model.to(DEVICE)
					val_loader = tasks[prev_task_id]['test']
					metrics = eval_single_epoch(model, val_loader, criterion, prev_task_id)
					log_metrics(metrics, time, prev_task_id)
					save_checkpoint(model, time)
					if prev_task_id == current_task_id:
						log_hessian(model, val_loader, time, prev_task_id)
					# save_checkpoint(model, time)
		# scheduler.step()
	end_experiment()
if __name__ == "__main__":
	run()