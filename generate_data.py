import sys

sys.path.append('./stylegan/')

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--number', help = 'how many to create', type = int)
parser.add_argument('--test', action = 'store_true', default = False)
parser.add_argument('--eval', action = 'store_true', default = False)
args = parser.parse_args()

def save_images(num_to_save, batch_size = 8, test = False):
    """
    save as many image samples as you would like
    set a batch size
    set whether it will go to test or train folder
    """
    added = ''
    if test == True:
      added = '_test'
    if not os.path.exists('latent_vectors{}'.format(added)):
        os.makedirs('latent_vectors{}'.format(added))
    if not os.path.exists('generated_images{}'.format(added)):
        os.makedirs('generated_images{}'.format(added))
    dnnlib.tflib.init_tf()
    with open('./stylegan/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    save_name = len(os.listdir('./latent_vectors{}'.format(added)))
    num_batches = num_to_save//batch_size
    last_batch_size = num_to_save % batch_size
    for batch_num in range(num_batches + 1):
        if batch_num == num_batches:
            batch_size = last_batch_size
            if last_batch_size == 0:
                continue
        if batch_num % 10 == 0:
            print('generating batch {} / {}'.format(batch_num, num_batches))
        latents = np.random.randn(batch_size, Gs.input_shape[1])
        src_latents = Gs.components.mapping.run(latents, None)
        style_latents = src_latents[:,0,:]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.components.synthesis.run(src_latents, randomize_noise=True, output_transform = fmt)
        for latent, image in zip(style_latents, images):
            # save image
            PIL.Image.fromarray(image, 'RGB').resize((256, 256)).save('generated_images{}/{}.jpeg'.format(added, save_name), 'JPEG')
            # save array
            np.save('latent_vectors{}/{}.npy'.format(added,save_name), latent)
            save_name += 1

def evaluate(eval_dir = 'eval_latent/'):
  dnnlib.tflib.init_tf()
  with open('./stylegan/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
    _G, _D, Gs = pickle.load(f)
  for vec in os.listdir(eval_dir):
    stl_vec = np.load(os.path.join(eval_dir, vec))
    lat_vec = np.tile(stl_vec, (1, 18, 1))
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.components.synthesis.run(lat_vec, randomize_noise=True, output_transform = fmt)
    image = images[0]
    PIL.Image.fromarray(image, 'RGB').resize((256, 256)).save('eval_photos/{}.jpeg'.format(os.path.splitext(vec)[0]), 'JPEG')

if __name__ == '__main__':
  if not args.eval:
    save_images(args.number, test =  args.test)
  else:
    if not os.path.exists('eval_photos'):
      os.makedirs('eval_photos')
    evaluate()


