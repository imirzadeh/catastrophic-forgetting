# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


def Xavier(m):
	if m.__class__.__name__ == 'Linear':
		fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
		std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
		a = math.sqrt(3.0) * std
		m.weight.data.uniform_(-a, a)
		m.bias.data.fill_(0.0)


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



# def conv3x3(in_planes, out_planes, stride=1):
# 	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
# 					 padding=1, bias=False)

# class BasicBlock(nn.Module):
# 	expansion = 1

# 	def __init__(self, in_planes, planes, stride=1):
# 		super(BasicBlock, self).__init__()
# 		self.conv1 = conv3x3(in_planes, planes, stride)
# 		self.bn1 = nn.BatchNorm2d(planes)
# 		self.conv2 = conv3x3(planes, planes)
# 		self.bn2 = nn.BatchNorm2d(planes)

# 		self.shortcut = nn.Sequential()
# 		if stride != 1 or in_planes != self.expansion * planes:
# 			self.shortcut = nn.Sequential(
# 				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
# 						  stride=stride, bias=False),
# 				#nn.BatchNorm2d(self.expansion * planes),
# 				# nn.Dropout(0.05),
# 			)

# 	def forward(self, x):
# 		out = relu(self.conv1(x)) #relu(self.bn1(self.conv1(x)))
# 		out = self.conv2(out)
# 		out += self.shortcut(x)
# 		out = relu(out)
# 		return out


# class ResNet(nn.Module):
# 	def __init__(self, block, num_blocks, num_classes, nf):
# 		super(ResNet, self).__init__()
# 		self.in_planes = nf

# 		self.conv1 = conv3x3(3, nf * 1)
# 		#self.bn1 = nn.BatchNorm2d(nf * 1)
# 		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
# 		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
# 		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
# 		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
# 		self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

# 	def _make_layer(self, block, planes, num_blocks, stride):
# 		strides = [stride] + [1] * (num_blocks - 1)
# 		layers = []
# 		for stride in strides:
# 			layers.append(block(self.in_planes, planes, stride))
# 			self.in_planes = planes * block.expansion
# 		return nn.Sequential(*layers)

# 	def forward(self, x, task_id):
# 		bsz = x.size(0)
# 		out = relu(self.conv1(x.view(bsz, 3, 32, 32)))
# 		out = self.layer1(out)
# 		out = self.layer2(out)
# 		out = self.layer3(out)
# 		out = self.layer4(out)
# 		out = avg_pool2d(out, 4)
# 		out = out.view(out.size(0), -1)
# 		out = self.linear(out)

# 		t = task_id
# 		offset1 = int((t-1) * 5)
# 		offset2 = int(t * 5)
# 		if offset1 > 0:
# 			out[:, :offset1].data.fill_(-10e10)
# 		if offset2 < 100:
# 			out[:, offset2:100].data.fill_(-10e10)
# 		return out


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
						  stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):
		out = relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf):
		super(ResNet, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
		self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
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

def ResNet18(nclasses=100, nf=20):
	return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
