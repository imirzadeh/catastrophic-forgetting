# Copyright 2017-present, Facebook, Inc.	
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d



def init_weights(m):
	# if type(m) == nn.Linear:
	# 	torch.nn.init.xavier_uniform(m.weight)
	# 	m.bias.data.fill_(0.01)
	if type(m) == nn.Conv2d:
		torch.nn.init.orthogonal_(m.weight)


class MLP(nn.Module):
	def __init__(self, hidden_layers, config, p=0.0):
		super(MLP, self).__init__()
		self.num_time_tensors = 50
		self.config = config
		self.W1 = nn.Linear(784, hidden_layers[0])
		self.relu = nn.ReLU(inplace=True)
		self.W2 = nn.Linear(hidden_layers[0], hidden_layers[1])
		self.W3 = nn.Linear(hidden_layers[1], hidden_layers[2])
		self.dropout_1 = nn.Dropout(p=config['dropout'])
		self.dropout_2 = nn.Dropout(p=config['dropout'])
		self.dropout_p = config(['dropout'])
		# if config['batchnorm'] > 0.0:
		# 	self.bn1 = nn.BatchNorm1d(hidden_layers[0], momentum=config['batchnorm'])
		# 	self.bn2 = nn.BatchNorm1d(hidden_layers[1], momentum=config['batchnorm'])


	def forward(self, x, task_id=None):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		# if self.config['batchnorm'] > 0.0:
		# 	out = self.bn1(out)
		# out = self.dropout_1(out)
		out = nn.functional.dropout(out, p=self.dropout_p)
		out = self.W2(out)
		out = self.relu(out)
		# if self.config['batchnorm'] > 0.0:
		# 	out = self.bn2(out)
		# out = self.dropout_2(out)
		out = nn.functional.dropout(out, p=self.dropout_p)
		out = self.W3(out)
		return out


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, config={}):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		# self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes)
		# self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
						  stride=stride, bias=False),
				# nn.BatchNorm2d(self.expansion * planes)
			)
		self.IC1 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

		self.IC2 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

	def forward(self, x):
		# out = relu(self.bn1(self.conv1(x)))
		# out = self.bn2(self.conv2(out))
		# out += self.shortcut(x)
		# out = relu(out)

		out = self.conv1(x)
		out = relu(out)
		out = self.IC1(out)

		out += self.shortcut(x)
		out = relu(out)
		out = self.IC2(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf, config={}):
		super(ResNet, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
		self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, task_id):
		bsz = x.size(0)
		out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		t = task_id
		offset1 = int((t-1) * 5)
		offset2 = int(t * 5)
		if offset1 > 0:
			out[:, :offset1].data.fill_(-10e10)
		if offset2 < 100:
			out[:, offset2:100].data.fill_(-10e10)
		return out

def ResNet18(nclasses=100, nf=20, config={}):
	net = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, config=config)
	if config.get('orthogonal-init', False):
		net.apply(init_weights)
	return net
