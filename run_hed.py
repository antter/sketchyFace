#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
from PIL import Image
import sys
import scipy.io as sio
import torchvision
import argparse

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'bsds500'
arguments_strIn = './generated_images'
arguments_strOut = './mat_output'

parser = argparse.ArgumentParser()
parser.add_argument('--test', help = 'whether to test', default = False, action = 'store_true')
args = parser.parse_args()

test = args.test

if test:
  add = '_test'
else:
  add = ''

arguments_strIn = arguments_strIn + add
arguments_strOut = arguments_strOut + add

if not os.path.exists(arguments_strOut):
    os.makedirs(arguments_strOut)






if not os.path.exists('./hed_output{}/0'.format(add)):
    os.makedirs('./hed_output{}/0'.format(add))

# end

##########################################################

class Network(torch.nn.Module):
  def __init__(self):
    super(Network, self).__init__()

    self.netVggOne = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggTwo = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggThr = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggFou = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netVggFiv = torch.nn.Sequential(
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False),
      torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=False)
    )

    self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
    self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

    self.netCombine = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
      torch.nn.Sigmoid()
    )

    self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items() })
  # end

  def forward(self, tenInput):
    tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
    tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
    tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

    tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

    tenVggOne = self.netVggOne(tenInput)
    tenVggTwo = self.netVggTwo(tenVggOne)
    tenVggThr = self.netVggThr(tenVggTwo)
    tenVggFou = self.netVggFou(tenVggThr)
    tenVggFiv = self.netVggFiv(tenVggFou)

    tenScoreOne = self.netScoreOne(tenVggOne)
    tenScoreTwo = self.netScoreTwo(tenVggTwo)
    tenScoreThr = self.netScoreThr(tenVggThr)
    tenScoreFou = self.netScoreFou(tenVggFou)
    tenScoreFiv = self.netScoreFiv(tenVggFiv)

    tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
    tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

    return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
  # end
# end

netNetwork = None

##########################################################

def estimate(tenInput):
  global netNetwork

  if netNetwork is None:
    netNetwork = Network().cuda().eval()
  # end

  # intWidth = tenInput.shape[2]
  # intHeight = tenInput.shape[1]

  # assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
  # assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

  return netNetwork(tenInput.cuda()).cpu()
  # .view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = os.listdir(main_dir)
        self.dir_name = main_dir
        self.transform = transform

    def __len__(self):
        return len(self.main_dir)

    def __getitem__(self, idx):
        img_loc = self.main_dir[idx]
        image = Image.open(os.path.join(self.dir_name, img_loc)).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, os.path.splitext(img_loc)[0]

################################################################

if __name__ == '__main__':
  
  transform = lambda x: torch.FloatTensor(numpy.ascontiguousarray(numpy.array(x.resize((256, 256), Image.BILINEAR))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
  dset = CustomDataSet(arguments_strIn, transform)
  loader = torch.utils.data.DataLoader(dset, num_workers = 4, batch_size = 32)
  j = 0
  for batch, filenames in loader:
    j += 1
    if j % 10 == 0:
      print('{} images completed'.format(j*32))
    tenOutput = estimate(batch)
    for i in range(len(batch)):
      to_print = tenOutput[i]
      to_print = numpy.squeeze(to_print).numpy()
      filenumber = os.path.splitext(filenames[i])[0]
      path = str(os.path.join(arguments_strOut, filenumber)) + '.mat'
      sio.savemat(path, {'predict': to_print})
      PIL.Image.fromarray(((1 - to_print)*255).astype(numpy.uint8)).save('hed_output{}/0/{}.png'.format(add, filenumber), 'png')
# end