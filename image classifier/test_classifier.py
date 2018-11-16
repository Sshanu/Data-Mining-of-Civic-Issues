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
import pickle
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

# optimizer = optim.SGD([p for p in detector.parameters() if p.requires_grad],
# 					  weight_decay=conf.l2, lr=conf.lr * conf.num_gpus * conf.batch_size, momentum=0.9)

# criterion = nn.CrossEntropyLoss()

for name in detector.state_dict():
	print(name)

ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])

def val_epoch():
	corrects = 0
	preds = []
	trues = []
	detector.eval()
	for val_b, batch in tqdm(enumerate(val_loader)):
		correct, pp, tt =  val_batch(batch) 
		trues.append(tt)
		preds.append(pp)
		corrects += correct

	print(corrects)
	print("val corrects", corrects)
	print("val size", len(trues)*conf.batch_size*conf.num_gpus)
	print("Accuracy validation: ", corrects/(len(trues)*conf.num_gpus*conf.batch_size)*100)
	print('-----------', flush=True)
	return corrects, preds, trues

def train_epoch():
	corrects = 0
	preds = []
	trues = []
	detector.eval()
	for train_b, batch in tqdm(enumerate(train_loader)):
		correct, pp, tt =  val_batch(batch) 
		trues.append(tt)
		preds.append(pp)
		corrects += correct

	print("train corrects", corrects)
	print("train size", len(trues)*conf.batch_size*conf.num_gpus)
	print("Accuracy train: ", corrects/(len(trues)*conf.num_gpus*conf.batch_size)*100)
	print('-----------', flush=True)
	return corrects, preds, trues

def val_batch(b):
	result = detector[b]
	scores = result.obj_scores
	labels = result.obj_labels
	_, preds = torch.max(scores.data,1)
	# print(scores, labels, preds)
	correct = torch.sum(preds == labels[:, 0].data)
	# print(correct)
	return correct, preds, labels[:, 0].data



print("Testing starts now!")
train_correct, train_preds, train_labels = train_epoch()
val_correct, val_preds, val_labels = val_epoch()
f = open("result_classifier", "wb")
pickle.dump([train_correct, train_preds, train_labels, val_correct, val_preds, val_labels], f)
