import os
import numpy as np
import pandas as pd
import skimage.io as io
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import cv2
import skimage


class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])
        print(inputName)
#         print(inputName, targetName)

        inputImage = io.imread(inputName)
        if (inputImage.shape[-1] == 4 ):
            inputImage = skimage.color.rgba2rgb(inputImage)
        if len(inputImage.shape) == 2:
            inputImage = inputImage[:, :, np.newaxis]
            inputImage = np.repeat(inputImage, 3, axis=-1)
            inputImage = inputImage[:, :, ::-1]
            inputImage = inputImage.astype(np.float32)
            inputImage -= np.array([104.00699 / 3, 116.66877 / 3, 122.67892 / 3])
            inputImage = inputImage.transpose((2, 0, 1))
        else:
            inputImage = inputImage[:, :, ::-1]
            inputImage = inputImage.astype(np.float32)
            inputImage -= np.array([104.00699, 116.66877, 122.67892])
            inputImage = inputImage.transpose((2, 0, 1))

        targetImage = io.imread(targetName)
        if len(targetImage.shape) == 3:
            targetImage = targetImage[:, :, 0]
        targetImage = targetImage > 0.0
        targetImage = targetImage.astype(np.float32)
        targetImage = np.expand_dims(targetImage, axis=0)

        inputImage = torch.Tensor(inputImage)
        targetImage = torch.Tensor(targetImage)
        return inputImage, targetImage

def _collate_fn(batch):
    
    minibatch_size = len(batch)
    size_x,size_y = 256,256
    imgs = torch.zeros(minibatch_size, 3, size_x, size_y)
    gts = torch.zeros(minibatch_size, size_x, size_y)
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


class ImageDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ImageDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
 
    

class TestDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
#         print(fi
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        fname = self.frame.iloc[idx, 0]
        inputName = os.path.join(self.rootDir, fname)

        inputImage = io.imread(inputName)[:, :, ::-1]
        K = 60000.0  # 180000.0 for sympascal
        H, W = inputImage.shape[0], inputImage.shape[1]
        sy = np.sqrt(K * H / W) / float(H)
        sx = np.sqrt(K * W / H) / float(W)
        inputImage = cv2.resize(inputImage, None, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        inputImage = inputImage.astype(np.float32)
        inputImage -= np.array([104.00699, 116.66877, 122.67892])
        inputImage = inputImage.transpose((2, 0, 1))

        inputImage = torch.Tensor(inputImage)
        return inputImage, fname, H, W

