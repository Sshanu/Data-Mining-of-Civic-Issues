"""
Data blob, hopefully to make collating less painful and MGPU training possible
"""
import numpy as np
import torch
from torch.autograd import Variable


class Blob(object):
    def __init__(self, mode='det', is_train=False, num_gpus=1, primary_gpu=0, batch_size_per_gpu=3):
        """
        Initializes an empty Blob object.
        :param mode: 'det' for detection and 'rel' for det+relationship
        :param is_train: True if it's training
        """
        assert mode in ('det', 'rel')
        assert num_gpus >= 1
        self.mode = mode
        self.is_train = is_train
        self.num_gpus = num_gpus
        self.batch_size_per_gpu = batch_size_per_gpu
        self.primary_gpu = primary_gpu

        self.imgs = []  # [num_images, 3, IM_SCALE, IM_SCALE] array
        self.im_sizes = []  # [num_images, 4] array of (h, w, scale, num_valid_anchors)

        self.gt_classes = []  # [num_gt,2] array of img_ind, class

        self.gt_sents = []
        self.gt_nodes = []
        self.sent_lengths = []

        self.batch_size = None

    @property
    def volatile(self):
        return not self.is_train

    def append(self, d):
        """
        Adds a single image to the blob
        :param datom:
        :return:
        """
        i = len(self.imgs)
        self.imgs.append(d['img'])

        h, w, scale = d['img_size']

        # all anchors
        self.im_sizes.append((h, w, scale))

        self.gt_classes.append(np.column_stack((        
            d['gt_classes'],
        )))


    def _chunkize(self, datom, tensor=torch.LongTensor):
        """
        Turn data list into chunks, one per GPU
        :param datom: List of lists of numpy arrays that will be concatenated.
        :return:
        """
        chunk_sizes = [0] * self.num_gpus
        for i in range(self.num_gpus):
            for j in range(self.batch_size_per_gpu):
                chunk_sizes[i] += datom[i * self.batch_size_per_gpu + j].shape[0]
        return Variable(tensor(np.concatenate(datom, 0)), volatile=self.volatile), chunk_sizes

    def reduce(self):
        """ Merges all the detections into flat lists + numbers of how many are in each"""
        if len(self.imgs) != self.batch_size_per_gpu * self.num_gpus:
            raise ValueError("Wrong batch size? imgs len {} bsize/gpu {} numgpus {}".format(
                len(self.imgs), self.batch_size_per_gpu, self.num_gpus
            ))

        self.imgs = Variable(torch.stack(self.imgs, 0), volatile=self.volatile)
        self.im_sizes = np.stack(self.im_sizes).reshape(
            (self.num_gpus, self.batch_size_per_gpu, 3))


        self.gt_classes, _ = self._chunkize(self.gt_classes)

    def _scatter(self, x, chunk_sizes, dim=0):
        """ Helper function"""
        if self.num_gpus == 1:
            return x.cuda(self.primary_gpu, async=True)
        return torch.nn.parallel.scatter_gather.Scatter.apply(
            list(range(self.num_gpus)), chunk_sizes, dim, x)

    def scatter(self):
        """ Assigns everything to the GPUs"""
        self.imgs = self._scatter(self.imgs, [self.batch_size_per_gpu] * self.num_gpus)

        self.gt_classes_primary = self.gt_classes.cuda(self.primary_gpu, async=True)

        # Predcls might need these
        self.gt_classes = self._scatter(self.gt_classes, [self.batch_size_per_gpu] * self.num_gpus)

    def __getitem__(self, index):
        """
        Returns a tuple containing data
        :param index: Which GPU we're on, or 0 if no GPUs
        :return: If training:
        (image, im_size, img_start_ind, anchor_inds, anchors, gt_boxes, gt_classes, 
        train_anchor_inds)
        test:
        (image, im_size, img_start_ind, anchor_inds, anchors)
        """
        if index not in list(range(self.num_gpus)):
            raise ValueError("Out of bounds with index {} and {} gpus".format(index, self.num_gpus))

        if index == 0 and self.num_gpus == 1:
            image_offset = 0
            if self.is_train:
                return (self.imgs, self.im_sizes[0], image_offset,
                        self.gt_classes)
            return self.imgs, self.im_sizes[0], image_offset, self.gt_classes

        image_offset = self.batch_size_per_gpu * index
        # TODO: Return a namedtuple
        if self.is_train:
            return (
            self.imgs[index], self.im_sizes[index], image_offset,
            self.gt_classes[index])
        return (self.imgs[index], self.im_sizes[index], image_offset,
               self.gt_classes[index])

