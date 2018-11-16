import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.pytorch_misc import enumerate_by_image, gather_nd, diagonal_inds, Flattener
from torchvision.models.vgg import vgg16
from torch.nn.parallel._functions import Gather


class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""

    def __init__(self, obj_labels=None, obj_scores=None):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([v is None for k, v in self.__dict__.items() if k != 'self'])


def gather_res(outputs, target_device, dim=0):
    """
    Assuming the signatures are the same accross results!
    """
    out = outputs[0]
    args = {field: Gather.apply(target_device, dim, *[getattr(o, field) for o in outputs])
            for field, v in out.__dict__.items() if v is not None}
    return type(out)(**args)


class ObjectDetector(nn.Module):
    """
    Core model for doing object detection + getting the visual features. This could be the first step in
    a pipeline. We can provide GT rois or use the RPN (which would then be classification!)
    """

    def __init__(self, classes, num_gpus=1):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param num_gpus: how many GPUS 2 use
        """
        super(ObjectDetector, self).__init__()

        self.classes = classes
        self.num_gpus = num_gpus

        vgg_model = load_vgg()
        self.features = vgg_model.features
        self.vgg_classifier = vgg_model.classifier
        output_dim = 4096
        self.score_fc = nn.Linear(output_dim, self.num_classes)

    @property
    def num_classes(self):
        return len(self.classes)

    def feature_map(self, x):
        """
        Produces feature map from the input image
        :param x: [batch_size, 3, size, size] float32 padded image
        :return: Feature maps at 1/16 the original size.
        Each one is [batch_size, dim, IM_SIZE/k, IM_SIZE/k].
        """
        return self.features(x)  # Uncomment this for "stanford" setting in which it's frozen:      .detach()
 
    def forward(self, x, im_sizes, image_offset,
                gt_classes=None):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        """
        fmap = self.feature_map(x)

        # Now classify them
        obj_states = self.vgg_classifier(fmap.view(fmap.size(0), -1))
        obj_scores = self.score_fc(obj_states)

        return Result(
            obj_labels=gt_classes,
            obj_scores=obj_scores
        )

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if any([x.is_none() for x in outputs]):
            assert not self.training
            return None
        return gather_res(outputs, 0, dim=0)


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
    model = vgg16(pretrained=pretrained)
    # del model.features._modules['30']  # Get rid of the maxpool
    del model.classifier._modules['6']  # Get rid of class layer
    for param in model.parameters():
        param.requires_grad = False
    return model
