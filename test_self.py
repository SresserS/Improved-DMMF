# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy, DM2FNet_woPhy_my, DM2FNet_my
from datasets import SotsDataset, OHazeDataset, SelfDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from metric import fmse, fciede

import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'

# exp_name = 'O-Haze'
exp_name = 'RESIDE_ITS'

args = {
    # 'snapshot': 'iter_20000_loss_0.05082_lr_0.000000', # baseline ohaze
    # 'snapshot': 'iter_20000_loss_0.05888_lr_0.000000', # my ohaze
    # 'snapshot': 'iter_40000_loss_0.01213_lr_0.000000', # baseline reside
    'snapshot': 'iter_40000_loss_0.01926_lr_0.000000', # my reside
} 

to_test = {
    'Self': 'data/Self'
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():

        for name, root in to_test.items():
            
            net = DM2FNet_my().cuda()
            dataset = SelfDataset(root)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            start_time = time.time()
            
            for idx, data in enumerate(dataloader):
                haze, fs = data

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()
                    

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"代码执行时间: {execution_time} 秒")


if __name__ == '__main__':
    main()
