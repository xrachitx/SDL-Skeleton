from PIL import Image
import math, os, time
import numpy as np
from torch.autograd import Variable
import torch
from tqdm import tqdm

class Trainer(object):
    # init function for class
    
    def __init__(self, network, optimizer, dataloader, args):
#     def __init__(self, network, optimizer, dataloader, args,batch_size):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.dataloader = dataloader
#         self.batch_size = batch_size

        if not os.path.exists('./weights/hed_sklarge/'):
            os.makedirs('./weights/hed_sklarge/')

        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        lossAcc = 0.0
        self.network.train()
        dataiter = iter(self.dataloader)
        for _ in range(self.args.resume_iter // self.args.lr_step):
            self.adjustLR()
        self.showLR()
        fp16=True
        scaler = torch.cuda.amp.GradScaler(enabled=True if fp16 else False) 
        for step in tqdm(range(self.args.resume_iter, self.args.max_step)):
            losses = []
            for _ in range(self.args.iter_size):
#                 try:
#                     data, target,dil = next(dataiter)
#                 except StopIteration:
#                     dataiter = iter(self.dataloader)
#                     data, target,dil = next(dataiter)
                try:
                    data, target = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    data, target= next(dataiter)
                    
#                 data, target,dil = data.cuda(self.args.gpu_id), target.cuda(self.args.gpu_id),dil.cuda(self.args.gpu_id)
#                 data, target,dil = Variable(data), Variable(target),Variable(dil)
                
                data, target = data.cuda(self.args.gpu_id), target.cuda(self.args.gpu_id)
                data, target = Variable(data), Variable(target)

#                 loss = self.network(data, target,dil,self.batch_size)
                with torch.cuda.amp.autocast(enabled=True if fp16 else False): # fp16 training
                    loss = self.network(data, target,True)
                
                try:
                    if np.isnan(float(loss.item())):
                        raise ValueError('loss is nan while training')
                    losses.append(loss)
                    lossAcc += loss.item()
                except:
                    print("avoiding")

            bLoss = torch.mean(torch.cat(losses))
            self.optimizer.zero_grad()
            scaler.scale(bLoss).backward()
#             bLoss.backward()
#             self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()

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
