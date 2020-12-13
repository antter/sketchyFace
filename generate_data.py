sys.path.append('./stylegan/')


import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys

def save_train_images(num_to_save, batch_size = 8):
    """
    save as many image samples as you would like
    set a batch size
    set add_on = True if some samples already exist
    """
    if not os.path.exists('latent_vectors'):
        os.makedirs('latent_vectors')
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    dnnlib.tflib.init_tf()
    with open('./stylegan/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    save_name = len(os.listdir('./latent_vectors'))
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
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        for latent, image in zip(latents, images):
            # save image
            PIL.Image.fromarray(image, 'RGB').resize((256, 256)).save('generated_images/{}.jpeg'.format(save_name), 'JPEG')
            # save array
            np.save('latent_vectors/{}.npy'.format(save_name), latent)
            save_name += 1
