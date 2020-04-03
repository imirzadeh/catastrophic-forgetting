from hessian_eigenthings.utils import progress_bar
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt


def parse_arguments():
	parser = argparse.ArgumentParser(description='Arg parser')
	parser.add_argument('--hidden_size', default=100, type=int, help='num hiddens')
	args = parser.parse_args()
	return args

def visualize_result(df, filename):
	ax = sns.lineplot(data=df,  dashes=False)
	ax.figure.savefig(filename, dpi=250)
	plt.close('all')

def get_full_hessian(loss_grad, model):
	# from https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
	cnt = 0
	loss_grad = list(loss_grad)
	for i, g in enumerate(loss_grad):
		progress_bar(
			i,
			len(loss_grad),
			"flattening to full gradient: %d of %d" % (i, len(loss_grad)),
		)
		g_vector = (
			g.contiguous().view(-1)
			if cnt == 0
			else torch.cat([g_vector, g.contiguous().view(-1)])
		)
		cnt = 1
	hessian_size = g_vector.size(0)
	hessian = torch.zeros(hessian_size, hessian_size)
	for idx in range(hessian_size):
		progress_bar(
			idx, hessian_size, "full hessian columns: %d of %d" % (idx, hessian_size)
		)
		grad2rd = torch.autograd.grad(
			g_vector[idx], model.parameters(), create_graph=True
		)
		cnt = 0
		for g in grad2rd:
			g2 = (
				g.contiguous().view(-1)
				if cnt == 0
				else torch.cat([g2, g.contiguous().view(-1)])
			)
			cnt = 1
		hessian[idx] = g2
	return hessian.cpu().data.numpy()
