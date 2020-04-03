from comet_ml import Experiment
import torch
import torch.nn as nn
import nni
from models import MLP
from ignite.metrics import Accuracy, Loss
from data_utils import get_permuted_mnist_tasks
from hessian_eigenthings import compute_hessian_eigenthings
from ignite.engine import Events, create_supervised_trainer,create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from comet_ml import Experiment
from pathlib import Path
import os
import copy
import numpy as np
import pandas as pd
from utils import get_full_hessian, visualize_result

config = nni.get_next_parameter()
config = {'epochs': 5, 'hiddens': 100, 'dropout': 0.0, 'batch_size': 128, 'lr': 0.1, }
TRIAL_ID = os.environ.get('NNI_TRIAL_JOB_ID', "UNKNOWN")
EXPERIMENT_DIRECTORY = './outputs/{}'.format(TRIAL_ID)
DEVICE = "cpu" if (not torch.cuda.is_available()) else "cuda:0"

# =============== SETTINGS ================
NUM_TASKS = 5
NUM_EIGENS = 10
EPOCHS = config['epochs']
HIDDENS = config['hiddens']
BATCH_SIZE = config['batch_size']
experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg",
						project_name="explore-hessian",
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
	experiment.log_metric('score', )
	experiment.log_asset_folder(EXPERIMENT_DIRECTORY)
	experiment.end()

def log_metrics(engine, time, task_id):
	metrics = engine.state.metrics
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
	criterion = torch.nn.CrossEntropyLoss()
	use_gpu = True if DEVICE != 'cpu' else False
	est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
		model,
		loader,
		criterion,
		num_eigenthings=NUM_EIGENS,
		power_iter_steps=10,
		power_iter_err_threshold=1e-5,
		momentum=0,
		use_gpu=use_gpu,
	)
	key = 'task-{}-epoch-{}'.format(task_id, time-1)
	hessian_eig_db[key] = est_eigenvals
	experiment.log_histogram_3d(name='task-{}-eigs'.format(task_id), step=time-1, values=est_eigenvals)

def save_checkpoint(model, time):
	filename = '{directory}/model-{time}.pth'.format(directory=EXPERIMENT_DIRECTORY, time=time)
	torch.save(model.cpu().state_dict(), filename)

def run():
	# basics
	model = MLP([HIDDENS, HIDDENS, 10], config = {'dropout': 0.0})
	tasks = get_permuted_mnist_tasks(NUM_TASKS, shuffle=True, batch_size=BATCH_SIZE)
	

	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
	criterion = nn.CrossEntropyLoss()
	trainer = create_supervised_trainer(model, optimizer, criterion, device=DEVICE)
	validator = create_supervised_evaluator(model,
											device=DEVICE,
											metrics={ 
											'accuracy': Accuracy(),
											'loss': Loss(criterion)})

	# hooks
	setup_experiment()

	# main loop
	time = 0
	for current_task_id in range(1, NUM_TASKS+1):
		print("========== TASK {} / {} ============".format(current_task_id, NUM_TASKS))
		train_loader =  tasks[current_task_id]['train']
		for epoch in range(1, EPOCHS+1):
			# train and save
			trainer.run(train_loader)
			time += 1
			save_checkpoint(model, time)

			# evaluate on all tasks up to now
			for prev_task_id in range(1, current_task_id+1):
				val_loader = tasks[prev_task_id]['test']
				validator.run(val_loader)
				log_metrics(validator, time, prev_task_id)
				log_hessian(model, val_loader, time, prev_task_id)
	end_experiment()
if __name__ == "__main__":
	run()