from PIL import Image
import math, os, time
import numpy as np
from torch.autograd import Variable
import torch
from tqdm import tqdm

class Trainer(object):
    # init function for class
    def __init__(self, network, optimizer, dataloader, args):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.dataloader = dataloader

        if not os.path.exists('weights'):
            os.makedirs('weights')

        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        lossAcc = 0.0
        self.network.train()
        dataiter = iter(self.dataloader)
        for _ in range(self.args.resume_iter // self.args.lr_step):
            self.adjustLR()
        self.showLR()
        for step in tqdm(range(self.args.resume_iter, self.args.max_step)):
            losses = []
            for _ in range(self.args.iter_size):
                try:
                    data, target,dil = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    data, target,dil = next(dataiter)

                data, target,dil = data.cuda(self.args.gpu_id), target.cuda(self.args.gpu_id),dil.cuda(self.args.gpu_id)
                data, target,dil = Variable(data), Variable(target),Variable(dil)

                loss = self.network(data, target,dil)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while training')
                losses.append(loss)
                lossAcc += loss.item()

            bLoss = torch.mean(torch.cat(losses))
            self.optimizer.zero_grad()
            bLoss.backward()
            self.optimizer.step()

            # adjust hed learning rate
            if (step > 0) and (step % self.args.lr_step) == 0:
                self.adjustLR()
                self.showLR()

            # visualize the loss
            if (step + 1) % self.args.disp_interval == 0:
                timestr = time.strftime(self.timeformat, time.localtime())
                print('{} iter={} loss={:<8.2f}'.format(
                    timestr, step + 1, lossAcc / self.args.disp_interval / self.args.iter_size))
                lossAcc = 0.0

            if (step + 1) % self.args.save_interval == 0:
                torch.save(self.network.state_dict(),
                           './weights/hed_sklarge/{}_{}.pth'.format(self.args.network, step + 1))

        torch.save(self.network.state_dict(), './weights/hed_sklarge/{}.pth'.format(self.args.network))

    def adjustLR(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.args.lr_gamma

    def showLR(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'], end=' ')
        print('')
