import pickle
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

info = json.load(open("data/stanford_filtered/cf_dicts.json", 'r'))
info['label_to_idx']['__background__'] = 0

class_to_ind = info['label_to_idx']
ind_to_class = dict((k, c) for c, k in class_to_ind.items())
print(ind_to_class)
ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])

f = open("result/result_classifier", "rb")
train_correct, train_preds, train_labels, val_correct, val_preds, val_labels = pickle.load(f)

train_pred = []
for pred in train_preds:
	for pr in pred:
		train_pred.append(pr)


train_label = []
for pred in train_labels:
	for pr in pred:
		train_label.append(pr)

val_pred = []
for pred in val_preds:
	for pr in pred:
		val_pred.append(pr)

val_label = []
for pred in val_labels:
	for pr in pred:
		val_label.append(pr)


print("train_correct", train_correct)
print("val_correct", val_correct)
print(len(train_pred))
print(len(train_label))
print(len(val_pred), len(val_label))

# label = train_label + val_label
# pred = train_pred + val_pred

# category_list = dict((i, []) for i in range(len(ind_to_classes)))

# new_cat = ['Fixing/Reparing Potholes', 'Construction of new footpaths',
#  'Flooding/Waterlogging Of Roads And Footpaths', 'Maintenance And Repair Of Manholes',
#  'Improve Storm Water Drains', 'Garbage', 'Maintenance of Roads and Footpaths - Others',
#  'Provide good driveable Roads', "Maintenance/Repair Of Streetlights", 'Repair of Existing Footpaths']
 
# ind2cat = dict((i+1, new_cat[i]) for i in range(len(new_cat)))

# for i in range(len(label)):
# 	category_list[label[i]].append(pred[i])

# print("--------------------")
# for k, cat in category_list.items():
# 	if cat == []:
# 		continue
# 	true = [k for i in range(len(cat))]
# 	print(ind2cat[k], len(cat), [(ind_to_class[j], count/len(cat)*100) for j, count in Counter(cat).most_common(5)])
# 	# print(ind_to_class[k], accuracy_score(true, cat)*100, f1_score(true, cat, average='macro')*100, len(cat))
# 	print("------")


category_list = dict((i, []) for i in range(len(ind_to_classes)))

for i in range(len(train_label)):
	category_list[train_label[i]].append(train_pred[i])
print("--------------------")
for k, cat in category_list.items():
	if cat == []:
		continue
	true = [k for i in range(len(cat))]
	print(ind_to_class[k], [(ind_to_class[j], count/len(cat)*100) for j, count in Counter(cat).most_common(5)])
	print(ind_to_class[k], accuracy_score(true, cat)*100, f1_score(true, cat, average='macro')*100, len(cat))
	print("------")

print("----------------------")
category_list = dict((i, []) for i in range(len(ind_to_classes)))
for i in range(len(val_label)):
	category_list[val_label[i]].append(val_pred[i])

for k, cat in category_list.items():
	if cat == []:
		continue
	true = [k for i in range(len(cat))]
	print(ind_to_class[k], [(ind_to_class[j], count/len(cat)*100) for j, count in Counter(cat).most_common(5)])
	print(ind_to_class[k], accuracy_score(true, cat)*100, f1_score(true, cat, average='macro')*100, len(cat))
	print("------")
print("----------------------")
# print("train", accuracy_score(train_label, train_pred)*100, f1_score(train_label, train_pred, average='macro')*100, len(train_pred))

# print("val", accuracy_score(val_label, val_pred)*100, f1_score(val_label, val_pred, average='macro')*100, len(val_label))
