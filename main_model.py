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
from torch.utils.data import Dataset, Sampler, DataLoader
import os
import PIL.Image
from numpy.random import choice
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', help = 'path of model checkpoint', default = None)
parser.add_argument('--frozen', help = 'whether vgg will be frozen', default= False, action = 'store_true')
parser.add_argument('--weight', help = 'weight to assign to most detailed sketch, must be >= 1, default off', default = 1, type = float)
parser.add_argument('--epochs', help = 'num epochs', default = 5, type = int)
parser.add_argument('--lr', help = 'learning rate', default = 0.001, type = float)
parser.add_argument('--save', help = 'name to save as', default = 'default_name')
parser.add_argument('--blur', help = 'whether to blur', default= False, action = 'store_true')
parser.add_argument('--eval', help = 'whether to evaluate the eval folder', default = False, action = 'store_true')

args = parser.parse_args()


# transforms_list = []
# if args.blur:
#   transforms_list.append(lambda x: cv2.GaussianBlur(x, (55, 55), 0))
# transforms_list.append(transforms.ToTensor())
# if args.frozen:
#   transforms_list.append(lambda x: x.repeat(3, 1, 1))
#   transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                               std=[0.229, 0.224, 0.225]))

# transform = transforms.Compose(transforms_list)

transform = transforms.ToTensor()
# transform = lambda x: torch.unsqueeze(tnsr(x), 0)

##################################################################


if not os.path.exists('./models'):
    os.mkdir('./models')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device))



#####################################################################

class trainSketchDataset(Dataset):
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

    def __getitem__(self, sampler):
        idx = next(sampler)
        sketch_fp = os.path.join(self.sketch_dir, idx[0], idx[1])
        fname = os.path.splitext(idx[1])[0]
        array_fp = os.path.join(self.array_dir, fname + '.npy')
        img = PIL.Image.open(sketch_fp)
        array = np.load(array_fp)
        return transform((np.array(img)/255).astype(np.float32)), array.astype(np.float32), idx[0], fname

class trainSketchSampler(Sampler):
    """
    samples according to schedule decided by sample_method argument
    """
    def __init__(self, sketch_dir, weight = 2):
        self.sketch_dir = sketch_dir
        sketch_levels = os.listdir(sketch_dir)
        sketch_levels = sorted(sketch_levels)
        self.sketch_levels = sketch_levels
        self.num_levels = len(sketch_levels)
        self.samples = [os.listdir(os.path.join(sketch_dir, level)) for level in self.sketch_levels]
        self.num_samples = np.array([len(level) for level in self.samples])
        self.weight_vector = np.linspace(weight, 1, num = self.num_levels)
        self.product = self.num_samples * self.weight_vector
        self.sum_ = sum(self.product)

    def __len__(self):
        return sum(self.num_samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self.sum_ < 1:
          raise StopIteration
        prob = self.product/self.sum_
        prob[prob < 0] = 0
        idx = choice(self.num_levels, p = prob)
        self.product[idx] = self.product[idx] - self.weight_vector[idx]
        self.sum_ = self.sum_ - self.weight_vector[idx]
        idx2 = choice(len(self.samples[idx]))
        ret2 = self.samples[idx].pop(idx2)
        ret1 = self.sketch_levels[idx]
        self.num_samples[idx] -= 1
        yield ret1, ret2


class testSketchDataset(Dataset):
    """
    meant to sample from only one level of sketch
    """
    def __init__(self, sketch_dir, array_dir):
        self.sketch_dir = sketch_dir
        self.array_dir = array_dir
        self.main_dir = os.listdir(sketch_dir)

    def __len__(self):
        return len(self.main_dir)

    def __getitem__(self, idx):
        img_loc = self.main_dir[idx]
        arr_loc = os.path.splitext(img_loc)[0] + '.npy'
        sketch_fp = os.path.join(self.sketch_dir, img_loc)
        array_fp = os.path.join(self.array_dir, arr_loc)
        img = PIL.Image.open(sketch_fp)
        array = np.load(array_fp)
        return transform((np.array(img)/255).astype(np.float32)), array

class evalDataset(Dataset):
    """
    meant to sample from only one level of sketch
    """
    def __init__(self, sketch_dir):
        self.sketch_dir = sketch_dir
        self.main_dir = os.listdir(sketch_dir)

    def __len__(self):
        return len(self.main_dir)

    def __getitem__(self, idx):
        img_loc = self.main_dir[idx]
        arr_loc = os.path.splitext(img_loc)[0] + '.npy'
        sketch_fp = os.path.join(self.sketch_dir, img_loc)
        img = PIL.Image.open(sketch_fp).convert('L')
        return transform(np.array(img.resize(256, 256))), arr_loc

###########################################################################



class asdf(nn.Module):
    
    
    def __init__(self):
        super(asdf, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, 4)
        # 64
        self.conv2 = nn.Conv2d(16, 64, 5, 2)
        # 32
        self.conv3 = nn.Conv2d(64, 128, 5, 2)
        # 16
        self.pool1 = nn.MaxPool2d(2, 2)
        # 8
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 6 * 6, 1028)
        # self.fc2 = nn.Linear
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool1(x)
        x = x.view(16 , -1)
        x = self.relu(self.fc1(x))
        return x
        
        
    
class FaceSketchModel(nn.Module):

    def __init__(self, vgg_load = True, frozen = False):
        """
        vgg_load: specify if you want to load pretrained vgg or if you plan on loading this model from a state_dict
        frozen: specify is vgg weights will be frozen
        """
        super(FaceSketchModel, self).__init__()
        self.frozen = frozen
        if vgg_load:
#             model = models.mobilenet_v2(pretrained = True)
            model = asdf()
            self.vgg16 = model
        else:
            self.vgg16 = models.mobilenet_v2(pretrained = False)
        if frozen:
            for param in self.vgg16.parameters():
                param.requires_grad = False
        self.last_layer = torch.nn.Linear(1028, 512)
        if not frozen:
            self.first_layer = nn.Conv2d(1, 3, kernel_size = (5,5), stride = (1,1), padding = (1, 1))
            self.relu = nn.ReLU()
            self.tform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def forward(self, x):
        if self.frozen:
            x = self.vgg16(x)
            x = self.last_layer(x)
            return x
        else:
            # x = self.first_layer(x)
            # x = self.relu(x)
            # x - self.tform(x)
            x = self.vgg16(x)
            x = self.last_layer(self.relu(x))
            return x

############################################################################


def train(model, dataloader, optimizer):
    """
    train the model!
    val_dataloaders should be a list of dataloaders in descending order of complexity of sketch
    """
    criterion = nn.MSELoss()
    model.to(device)
    all_losses = []
    num_batches = 0
    train_running_loss = 0.0
    batches_loss = 0
    listt = np.array(0)
    listtt = np.array(0)
    for i, data in enumerate(dataloader):
        num_batches += 1
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        listt = np.append(listt, loss.item())
        batches_loss += loss.item()
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        listtt = np.append(listtt, np.var(np.array(output.detach().cpu()), axis = 0).mean())
        if i % 199 == 0 and i != 0:
            print('loss over 200 batches: {}'.format(batches_loss / 200))
            print('loss std over 200 batches: {}'.format(listt.var()**(1/2) ))
            print('mean prediction std over 200 batches {}'.format(listtt.mean()))
            print('')
            batches_loss = 0
            lisst = np.array(0)
    return all_losses, train_running_loss/num_batches

def test(model, dataloader, j = 0):
    """
    test model, output different loss for each category
    """
    model.float()
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    num_batches = 0
    variances = np.array(0)
    for i, data in enumerate(dataloader):
      num_batches += 1
      data, target = data[0].to(device), data[1].to(device)
      output = model(data)
      loss = criterion(output, target)
      variances = np.append(variances, output.detach().cpu())
      val_loss += loss.item()
    mean_loss = val_loss / num_batches
    print('VAL LOSS for {} : {}'.format(j, mean_loss))
    print('VAL MEAN VARIANCE for {} : {}'.format(j, variances.mean()))
    
    return mean_loss


def trainer(model,
          num_epochs = 10,
          lr = 0.001,
          save_name = 'model',
          sketch_dir = './hed_output',
          latent_dir = './latent_vectors',
          batch_size = 16,
          weight = 2):
  optimizer = optim.Adam(model.parameters(), lr=lr)
  min_validation_loss = np.inf
  all_losses = []
  all_val_losses = []
  for Q in range(num_epochs):
    model.train()
    train_dset = trainSketchDataset(sketch_dir, latent_dir)
    sampler = trainSketchSampler(sketch_dir, weight = weight)
    train_dloader = DataLoader(dataset= train_dset,sampler= sampler, batch_size = batch_size, shuffle= False)
    levels = os.listdir(sketch_dir)
    levels = sorted(levels)
    val_dloaders = []
    for level in levels:
      level_path = os.path.join(sketch_dir + '_test', level)
      array_path = os.path.join(latent_dir + '_test')
      val_dset = testSketchDataset(level_path, array_path)
      val_dloader = DataLoader(dataset = val_dset, batch_size = batch_size, shuffle = False)
      val_dloaders.append(val_dloader)
    ret1, ret2 = train(model, train_dloader, optimizer)
    all_losses += ret1
    model.eval()
    print('\n')
    print(f'Loss after {Q + 1} epochs: {ret2}')
    print('\n')
    val_losses = []
    for i, val_dataloader in enumerate(val_dloaders):
      val_losses.append(test(model, val_dataloader, i))
    val_losses = np.array(val_losses)
    all_val_losses.append(val_losses)
    print('\n')
    print('TOTAL VAL LOSS at EPOCH {}: {}'.format(Q + 1, val_losses.mean()))
    print('\n')
    if val_losses.mean() < min_validation_loss:
      torch.save(model.state_dict(), os.path.join('./models/', save_name))
      min_validation_loss = val_losses.mean()
    else:
      return all_losses, all_val_losses
  return all_losses, all_val_losses
    

#################################################################################################

if __name__ == '__main__':
  sketch_dir = 'hed_output_test/'
  latent_dir = 'latent_vectors/'
  eval_dir = 'eval/'
  model = FaceSketchModel(frozen = args.frozen)
  if args.ckpt != None:
    model.load_state_dict(torch.load(args.ckpt))
  if not args.eval: 
    trainer(model, num_epochs = args.epochs,
              lr = args.lr,
              save_name = args.ckpt if args.ckpt != None else args.save,
              weight = args.weight)
  else:
    if not os.path.exists('eval_latent/'):
      os.mkdir('eval_latent/')
    dset = evalDataset('eval/')
    dloader = DataLoader(dset, batch_size = 8)
    model.eval()
    for batch, save_loc in dloader:
      output = model(batch).cpu().detach().numpy()
    for out, arr_loc in zip(output, save_loc):
      np.save(os.path.join('eval_latent/', arr_loc), out)
  

