import numpy as np
import cv2
import random
import pandas as pd
import skimage.io as io
import torch
import os
from torch.utils.data import Dataset, DataLoader


def _collate_fn(batch):
    
    minibatch_size = len(batch)
    size_x,size_y = 256,256
    imgs = torch.zeros(minibatch_size, 3, size_x, size_y)
    fluxes = torch.zeros(minibatch_size, 2, size_x, size_y)
    dilmasks = torch.zeros(minibatch_size, 1, size_x, size_y)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        w_a,h_a = tensor.shape[1], tensor.shape[2]
        maxx = max(w_a,h_a)
        cap = 256
        extra = 0

        if maxx<= cap:
            new_w, new_h = w_a,h_a
        else:
            r = cap/maxx
            new_w, new_h = int(w_a*r), int(h_a*r)
            img_transform = transforms.Compose([transforms.ToPILImage(mode="RGB"),transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            gt_transform = transforms.Compose([transforms.ToPILImage(mode="L"),transforms.Resize((new_h, new_w)), transforms.ToTensor()])
            tensor = img_transform(tensor)
            target = gt_transform(target)

        p_up = (cap+extra-new_h)//2
        p_down = cap+extra-p_up-new_h

        p_left = (cap+extra-new_w)//2
        p_right = cap+extra-p_left-new_w
            
        tensor = torch.nn.functional.pad(tensor,(p_left,p_right,p_up,p_down),value=0)
        target = torch.nn.functional.pad(target,(p_left,p_right,p_up,p_down),value=0)
        target = torch.squeeze(target,axis=0)
        imgs[x,:,:,:] = tensor
        gts[x,:,:] = target
        
    
#     targets = torch.IntTensor(targets)
    return imgs, gts

class DataLayer(Dataset):

    def __init__(self, fileNames,rootDir):
        # data layer config
        self.rootDir = rootDir
#         self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)
        self.mean = np.array([103.939, 116.779, 123.68])

        # read filename list for each dataset here
#         self.fnLst = open(self.data_dir + 'SKLARGE/train_pair_255_s_all.lst').readlines()
        # randomization: seed and pick

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        # load image, flux and dilmask
        
        self.image, self.flux, self.dilmask = self.loadsklarge(idx,
                                                               idx)
        return self.image, self.flux, self.dilmask

    def loadsklarge(self, imgidx, gtidx):
        # load image and skeleton
        inputName = os.path.join(self.rootDir, self.frame.iloc[imgidx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[gtidx, 1])
        image = cv2.imread(inputName, 1)
        skeleton = cv2.imread(targetName, 0)
        print("IMAGE: ", image.shape, "skeleton: ", skeleton.shape)
        skeleton = (skeleton > 0).astype(np.uint8)
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2, 0, 1))

        # compute flux and dilmask
        kernel = np.ones((15, 15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1 - skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                      labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff ** 2, axis=0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0, rev > 0] = np.divide(diff[0, rev > 0], dist[rev > 0])
        direction[1, rev > 0] = np.divide(diff[1, rev > 0], dist[rev > 0])

        direction[0] = direction[0] * (dilmask > 0)
        direction[1] = direction[1] * (dilmask > 0)

        flux = -1 * np.stack((direction[0], direction[1]))

        dilmask = (dilmask > 0).astype(np.float32)
        dilmask = dilmask[np.newaxis, ...]
        print(image.shape,flux.shape,dilmask.shape)
        return image, flux, dilmask


class TestDataLayer(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = sorted(os.listdir(self.rootDir))
        print(type(self.frame))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        fname = self.frame[idx]
        inputName = os.path.join(self.rootDir, fname)

        inputImage = io.imread(inputName)[:, :, ::-1]
        inputImage = inputImage.astype(np.float32)
        inputImage -= np.array([104.00699, 116.66877, 122.67892])
        inputImage = inputImage.transpose((2, 0, 1))
        inputImage = torch.Tensor(inputImage)
        return inputImage, fname

