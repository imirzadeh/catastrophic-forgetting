from data_utils import get_mnist_tasks
import os
import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



def save_embeddings():
	TASK_SIZE = 3000
	data = get_mnist_tasks(num_tasks=3, shuffle=True, batch_size=TASK_SIZE)
	task_1 = data[1]
	task_2 = data[2]
	task_3 = data[3]

	task_1_images, task_1_targets = [], []
	task_2_images, task_2_targets = [], []
	task_3_images, task_3_targets = [], []

	for batch_idx, (data, target) in enumerate(task_1['train']):
		
			# print(target.numpy())
			# task_1_images += data.numpy().reshape(50, 784)
			task_1_images = data.numpy().reshape(TASK_SIZE, 784)
			task_1_targets = target.numpy().reshape(TASK_SIZE)
			break

	for batch_idx, (data, target) in enumerate(task_2['train']):
		
			# print(target.numpy())
			# task_1_images += data.numpy().reshape(50, 784)
			task_2_images = data.numpy().reshape(TASK_SIZE, 784)
			task_2_targets = target.numpy().reshape(TASK_SIZE)
			break

	for batch_idx, (data, target) in enumerate(task_3['train']):
		
			# print(target.numpy())
			# task_1_images += data.numpy().reshape(50, 784)
			task_3_images = data.numpy().reshape(TASK_SIZE, 784)
			task_3_targets = target.numpy().reshape(TASK_SIZE)
			break

	# task_2_targets = np.vectorize(lambda x: x+10)(task_2_targets)
	# task_3_targets = np.vectorize(lambda x: x+20)(task_3_targets)


	images = np.concatenate((task_1_images, task_2_images, task_3_images), axis=0)
	images = np.concatenate((task_1_images, task_2_images), axis=0)

	targets = np.append(task_1_targets, task_2_targets)
	# targets = np.append(targets, task_3_targets)

	# images = task_1_images
	# targets = task_1_targets
	embeddings = TSNE(n_jobs=4).fit_transform(images)
	vis_x = embeddings[:, 0]
	vis_y = embeddings[:, 1]
	plt.scatter(vis_x, vis_y, c=targets, cmap=plt.cm.get_cmap("jet", 10), marker='.')
	plt.colorbar(ticks=range(10))
	plt.clim(-0.1, 10.1)
	plt.show()
	plt.savefig('5k-2tasks-no-class-aug.png', dpi=300)



if __name__ == "__main__":
	save_embeddings()