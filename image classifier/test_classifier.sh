#!/usr/bin/env bash
# Train the model without COCO pretraining
export CUDA_VISIBLE_DEVICES=0,2

python models/test_classifier.py -b 24 -lr 1e-3 -ckpt checkpoints/classifier1/classifier-49.tar -test -nepoch 50 -ngpu 2 -nwork 2 -p 100 -clip 5

# If you want to evaluate on the frequency baseline now, run this command (replace the checkpoint with the
# best checkpoint you found).
#export CUDA_VISIBLE_DEVICES=0
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-24.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=1
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=2
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#
#
