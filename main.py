import os
import nni
import numpy as np
import pandas as pd
from comet_ml import Experiment
import torch
import torch.nn as nn
from models import MLP
from data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks
from hessian_eigenthings import compute_hessian_eigenthings
from pathlib import Path
from utils import visualize_result

config = nni.get_next_parameter()
# config = {'epochs': 5, 'hiddens': 100, 'dropout': 0.5,
# 		 'batch_size': 64, 'lr': 0.1, 'gamma': 0.9,
# 		 'batchnorm': 0.0, 'momentum': 0.8}

TRIAL_ID = os.environ.get('NNI_TRIAL_JOB_ID', "UNKNOWN")
EXPERIMENT_DIRECTORY = './outputs/{}'.format(TRIAL_ID)
DEVICE = 'cuda'

# =============== SETTINGS ================
NUM_TASKS = 5
NUM_EIGENS = 1
EPOCHS = config['epochs']
HIDDENS = config['hiddens']
BATCH_SIZE = config['batch_size']
experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg",
						project_name="neurips-5tasks-perm",
						auto_param_logging=False, auto_metric_logging=False,
						workspace="nn-forget", disabled=False)

loss_db = {t:[10 for i in range(NUM_TASKS*EPOCHS)] for t in range(1, NUM_TASKS+1)}
acc_db = {t:[0 for i in range(NUM_TASKS*EPOCHS)] for t in range(1, NUM_TASKS+1)}
hessian_eig_db = {}


def setup_experiment():
	print('Experiment started')
	experiment.log_parameters(config)
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

def log_hessian(model, loader, time, task_id):
	criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
	use_gpu = True if DEVICE != 'cpu' else False
	est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
		model,
		loader,
		criterion,
		num_eigenthings=NUM_EIGENS,
		power_iter_steps=12,
		power_iter_err_threshold=1e-5,
		momentum=0,
		use_gpu=True,
	)
	key = 'task-{}-epoch-{}'.format(task_id, time-1)
	hessian_eig_db[key] = est_eigenvals
	experiment.log_histogram_3d(name='task-{}-eigs'.format(task_id), step=time-1, values=est_eigenvals)

def save_checkpoint(model, time):
	filename = '{directory}/model-{trial}-{time}.pth'.format(directory=EXPERIMENT_DIRECTORY, trial=TRIAL_ID, time=time)
	torch.save(model.cpu().state_dict(), filename)

def train_single_epoch(net, optimizer, loader, criterion):
	net = net.to(DEVICE)
	net.train()
	
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		pred = net(data)#net(data, task_id)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net

def eval_single_epoch(net, loader, criterion):
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			output = net(data)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}

def run():
	# basics
	model = MLP([HIDDENS, HIDDENS, 10], config=config).to(DEVICE)
	tasks = get_rotated_mnist_tasks(NUM_TASKS, shuffle=True, batch_size=BATCH_SIZE)
	
	# optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['gamma'])
	criterion = nn.CrossEntropyLoss().to(DEVICE)


	# hooks
	setup_experiment()
	# main loop
	time = 0
	for current_task_id in range(1, NUM_TASKS+1):
		print("========== TASK {} / {} ============".format(current_task_id, NUM_TASKS))
		train_loader =  tasks[current_task_id]['train']
		task_lr = config['lr']*(config['gamma']**(current_task_id-1))
		optimizer = torch.optim.SGD(model.parameters(), lr=task_lr, momentum=config['momentum'])
		for epoch in range(1, EPOCHS+1):
			# train and save
			train_single_epoch(model, optimizer, train_loader, criterion)
			time += 1

			# evaluate on all tasks up to now
			for prev_task_id in range(1, current_task_id+1):
				model = model.to(DEVICE)
				val_loader = tasks[prev_task_id]['test']
				metrics = eval_single_epoch(model, val_loader, criterion)
				log_metrics(metrics, time, prev_task_id)
				if epoch == EPOCHS:
					log_hessian(model, val_loader, time, prev_task_id)
					save_checkpoint(model, time)
		# scheduler.step()
	end_experiment()
if __name__ == "__main__":
	run()