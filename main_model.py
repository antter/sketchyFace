import torch
import torchvision
import torchvision.models as models
import numpy as np
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, Sampler
import os
import PIL.Image
from np.random import choice
import argparse

##################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', help = 'path of model checkpoint', default = None)
parser.add_argument('--frozen', help = 'whether vgg will be frozen', default= True)
parser.add_argument('--weight', help = 'weight to assign to most detailed sketch, must be > 1, default off', default = 1)
parser.add_argument('--epochs', help = 'num epochs', default = 2)
parser.add_argument('--lr', help = 'learning rate', default = 0.001)
parser.add_argument('--save', help = 'directory to save', default = './models')

if os.path.exists(parser.save):
    os.mkdir(parser.save)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# transformations for input images

transforms_list = []

transforms_list.append(lambda x: cv2.GaussianBlur(x, (55, 55), 0))
transforms_list.append(transforms.ToTensor())
if parser.frozen:
    transforms_list.append(lambda x: x.repeat(3, 1, 1))
transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))

transform = transforms.Compose(transforms_list)


##############################################################
class sketchDataset(Dataset):
    """
    getitem returns a sketch, array, and filename
    idx will be a list of [sketch_level, filename]
    """
    def __init__(self, sketch_dir, array_dir):
        self.sketch_dir = sketch_dir
        self.array_dir = array_dir
        self.num_sketch_levels = len(os.listdir(sketch_dir))

    def __len__(self):
        return len(os.listdir(self.array_dir))*self.num_sketch_levels

    def __getitem__(self, idx):
        sketch_fp = os.path.join(self.sketch_dir, idx[0], idx[1])
        array_fp = os.path.join(self.array_dir, idx[1])
        img = PIL.Image.open(sketch_fp)
        array = np.load(array_fp)
        return transform(np.array(img)), array, idx[0], idx[1]

class sketchSampler(Sampler):
    """
    samples according to schedule decided by sample_method argument
    """
    def __init__(self, sketch_dir, weight = 2):
        self.sketch_dir = sketch_dir
        sketch_levels = os.listdir(os.listdir(sketch_dir)
        self.sketch_levels = sorted(sketch_levels)
        self.num_levels = len(sketch_levels)
        self.samples = [os.listdir(os.path.join(sketch_dir, level)) for level in self.sketch_levels]
        self.num_samples = np.array(len(level) for level in self.samples)
        self.weight_vector = np.linspace(weight, 1, num = self.num_levels)
        self.product = self.num_samples * self.weight_vector
        self.sum = sum(self.product)

    def __len__(self):
        return sum(self.num_samples)

    def __iter__(self):
        return self

    def __next__(self):
        prob = self.product/self.sum
        idx = choice(self.num_levels, prob)
        self.product[idx] = self.product[idx] - self.weight_vector[idx]
        self.sum = self.sum - self.weight_vector[idx]
        idx2 = choice(self.num_samples[idx])
        ret1 = self.sketch_levels[idx]
        ret2 = self.samples[idx][idx2]
        self.num_samples[idx] -= 1
        yield ret1, ret2

###########################################################################


class sketchToLatent(nn.Module)

    def __init__(self, vgg_load = True, frozen = True):
        """
        vgg_load: specify if you want to load pretrained vgg or if you plan on loading this model from a state_dict
        frozen: specify is vgg weights will be frozen
        """
        self.frozen = frozen
        if vgg_load:
            self.vgg16 = models.vgg16(pretrained=True)
        else:
            self.vgg16 = models.vgg16()
        if frozen:
            for param in vgg16.features.parameters():
                param.requires_grad = False
        vgg16.classifier[6] = torch.nn.Linear(4096, 512)
        if not frozen:
            self.first_layer = F.Conv2d(1, 3, kernel_size = (3,3), stride = (1,1), padding = (1, 1))
            self.relu = nn.ReLU()

    def forward(self, x):
        if self.frozen:
            return self.vgg16(x)
        else:
            x = self.first_layer(x)
            x = self.relu(x)
            return self.vgg16(x)

############################################################################


def train(model, dataloader):
    """
    train the model!
    """
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=parser.lr, momentum=0.9)
    criterion = nn.MSE()
    model.train()
    model.to(device)
    train_running_loss = 0.0
    all_losses = []
    for j in range(parser.epochs)):
        batches_loss = 0
        for i, data in enumerate(dataloader):
            data, target = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_running_loss += loss.item()
            batches_loss += loss.item()
            all_losses.append()
            loss.backward()
            optimizer.step()
            if i % 9 == 0:
                print('loss over 10 batches: {}'.format(batches_loss / 10))
                batches_loss = 0
        train_running_loss = 0.0
        print(f'Loss after {j + 1} epochs: {train_running_loss / len(train_dataloader)}')


def test(model, dataloader):
    """
    test model, output different loss for each category
    """