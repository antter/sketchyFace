#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.append('./face-parsing.PyTorch')

from logger import setup_logger
from model import BiSeNet

import torch
import time
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--test', help = 'whether or not to test', action = 'store_true', default = False)
args = parser.parse_args()

def evaluate(dspth='./generated_images', cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('./face-parsing.PyTorch', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    dset = CustomDataSet(dspth)
    loader = torch.utils.data.DataLoader(dset, num_workers = 4, batch_size = 32)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with torch.no_grad():
      i = 0
      for images, filenames in loader:
        i += 1
        batch = normalize(images)
        batch = batch.cuda()
        out = net(batch)[0]
        images = images.permute(0, 2, 3, 1)
        if i % 30 == 0:
          print("{} images done".format(i*32))
        for image, output, filename in zip(images, out, filenames):
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
            img = image.numpy()
            img[parsing == 0] = 0
            im = Image.fromarray((img*255).astype(np.uint8))
            im.save(os.path.join(dspth,'{}.jpeg'.format(filename)))


class CustomDataSet(Dataset):
    def __init__(self, dir_name):
        self.main_dir = os.listdir(dir_name)
        self.dir_name = dir_name
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.main_dir)

    def __getitem__(self, idx):
        img_loc = self.main_dir[idx]
        image = Image.open(os.path.join(self.dir_name, img_loc)).convert("RGB")
        return self.to_tensor(image), os.path.splitext((img_loc))[0]




if __name__ == "__main__":
    path = './generated_images'
    if args.test:
      path += '_test'
    evaluate(dspth=path, cp='79999_iter.pth')


