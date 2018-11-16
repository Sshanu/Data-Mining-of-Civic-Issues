"""
Training script 4 Detection
"""
from dataloaders.classifier_data import VGDataLoader, VG
from lib.classifier import ObjectDetector
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, IM_SCALE
from torch.nn import functional as F
import torch.nn as nn
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from tqdm import tqdm

cudnn.benchmark = True
conf = ModelConfig()

train, val, _ = VG.splits(num_val_im=conf.val_size)
train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=conf.batch_size,
											   num_workers=conf.num_workers,
											   num_gpus=conf.num_gpus)

detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus)
detector.cuda()

# Note: if you're doing the stanford setup, you'll need to change this to freeze the lower layers

optimizer = optim.SGD([p for p in detector.parameters() if p.requires_grad],
					  weight_decay=conf.l2, lr=conf.lr * conf.num_gpus * conf.batch_size, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for name in detector.state_dict():
	print(name)

start_epoch = -1
if conf.ckpt is not None:
	ckpt = torch.load(conf.ckpt)
	if optimistic_restore(detector, ckpt['state_dict']):
		start_epoch = ckpt['epoch']

def train_epoch(epoch_num):
	detector.train()
	tr = []
	start = time.time()
	for b, batch in tqdm(enumerate(train_loader)):
		tr.append(train_batch(batch))

		if b % conf.print_interval == 0 and b >= conf.print_interval:
			mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
			time_per_batch = (time.time() - start) / conf.print_interval
			print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
				epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
			print(mn)
			print('-----------', flush=True)
			start = time.time()
	return pd.concat(tr, axis=1)

def train_batch(b):
	"""
	:param b: contains:
		  :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
		  :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
		  :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
								  RPN feature vector that give us all_anchors,
								  each one (img_ind, fpn_idx)
		  :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

		  :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

		  Training parameters:
		  :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
									be used to compute the training loss (img_ind, fpn_idx)
		  :param gt_boxes: [num_gt, 4] GT boxes over the batch.
		  :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)

	:return:
	"""
	result = detector[b]
	scores = result.obj_scores
	labels = result.obj_labels

	# detector loss
	loss = criterion(scores, labels[:, 0])           
	res = pd.Series([loss.data[0]],
						['loss'])

	optimizer.zero_grad()
	loss.backward()
	clip_grad_norm(
		[(n, p) for n, p in detector.named_parameters() if p.grad is not None],
		max_norm=conf.clip, clip=True)
	optimizer.step()

	return res

def val_epoch():
	corrects = 0
	tr = []
	trues = []
	detector.eval()
	for val_b, batch in enumerate(val_loader):
		correct, tt, ll =  val_batch(batch) 
		tr.append(tt)
		corrects += correct
		trues.append(ll)
	print(pd.concat(tr, axis=1).mean(1))
	print("corrects", corrects)
	print("val size", conf.val_size)
	print("size", len(trues)*conf.batch_size*conf.num_gpus)
	print("Accuracy: ", corrects/(len(trues)*conf.num_gpus*conf.batch_size)*100)
	print('-----------', flush=True)



def val_batch(b):
	result = detector[b]
	scores = result.obj_scores
	labels = result.obj_labels
	_, preds = torch.max(scores.data,1)
	loss = criterion(scores, labels[:, 0])           
	res = pd.Series([loss.data[0]],
						['loss'])

	return torch.sum(preds == labels[:, 0].data), res, labels[:, 0].data



print("Training starts now!")
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
	rez = train_epoch(epoch)
	print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['loss'], rez.mean(1)), flush=True)
	val_epoch()
	torch.save({
		'epoch': epoch,
		'state_dict': detector.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, os.path.join(conf.save_dir, '{}-{}.tar'.format('classifier', epoch)))
