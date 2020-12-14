# sketchyFace
### a method for creating realistic faces from simple sketches

This is based off StyleGAN, taking some ideas from SketchyGAN.

The overall idea is that we can use StyleGAN to create paired images and style vectors. We can then process these images to make them look like sketches. We then train a convolutional network to predict the style vector from the sketch. 

The model is outlined below. As our ConvNet we use a very basic archtiecture with 3 Conv layers and 2 FC. This can be changed in main_model.py.

![Methodology](https://github.com/antter/sketchyFace/blob/main/method.png)

Our baseline method's results are presented below. This baseline does only minimal preprocessing and thus is unsuitable for actual rough sketches, they would have to look as detailed as in the examples. Results other than the baseline will come soon.

![Results](https://github.com/antter/sketchyFace/blob/main/results.png)



### Quick explanation:
  
  1. generate_images creates a paired face image and style vector dataset
  2. filter_images will filter the background out of the face images
  3. run_hed will create matlab files for postprocessing and save HED images, creating the first level of sketch
  4. matlab_process/Postrocess.m will then create simpler sketches from the HED, one can generate as many levels of sketch as they want
  5. main_model can be used for training on the dataset, with some options
  6. sketches can then be uploaded to a directory named /eval, then main_model --eval can be run to generate vectors, then generate_images --eval to generate images

### Instructions

You will need to download the StyleGAN repo and the face-parsing.Pytorch repo

```
git clone https://github.com/NVlabs/stylegan.git
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
```

You will then need to install the pretrained StyleGAN model and put it in the stylegan directory, and you will need to download the pretrained face-parsing model and put it in that directory.

In order to generate N face samples and paired style vectors, run

```
python generate_images.py --number N
```

This will create a generated_images and latent_vectors directory, where the generated data is stored under the same filename.

To filter out the background of the generated images run 

```
python filter_images.py
```

To create .mat files for further postprocessing run

```
python run_hed.py
```

This will save .mat files in a mat_output directory, and HED edge photos (first level of sketch) in a hed_output directory.

If you wish to obtain simpler sketch images, change directories to /matlab_process, and run Postprocessing.m.

For instructions on Postprocessing.m requirements consult https://github.com/phillipi/pix2pix/blob/master/scripts/edges/PostprocessHED.m. Piotr's image toolbox must be installed. To see how to work this in colab consult guide.ipynb

A snippet looks like:
```
! octave --eval 'feval ("PostprocessHED", "../mat_output/", "../hed_output/1", 256, 25.0/255.0, 10)'
```
The correct directory must be specified, and the 25/255 parameter can be adjusted to create "sketchier" sketches. If you wish to have varying levels of complexity of sketches, it is important that you put them in the hed_output/n directory, with larger n for simpler sketches. Run this code for as many levels of sketch complexity you wish to have. hed_output/0 represents the output of run_hed, the most complex sketch option.

To collect the test dataset, simply rerun all the same code but add --test at the end of the python command line calls. The directories will be named the same, with \_test appended to the end of them. The matlab calls will be the same, just change the directory names to include \_test.

To train the model run main_model.py. There are some options here.

```
python main_model.py --save SAVENAME_OF_MODEL --ckpt RESUME_TRAINING_FROM --epochs NUM_EPOCHS --lr LR --blur --weight WEIGHT
```

--blur is for applying Gaussian blurring as pre-processing before the model, and WEIGHT is for scheduling training to go from complex to simple sketches. A weight of e.g. 2 will make it so a sketch from hed_output/0 is twice as likely to be picked for the batch, proportional to the amount of sketches left to be trained on. That is, when 1/3 of hed_output/0 and 2/3 of hed_output/n are remaining to be trained on, they have an equal chance of being sampled from next. The weights for sketch level in between are linear from 1 to WEIGHT. Leaving the weight as 1 or not putting the argument will result in random sampling from all sketch levels.

Once a model has been trained, latent vectors can be predicted from photos in the eval directory. Run

```
python main_model.py --ckpt CKPT  --blur --eval
```
It is important that if you put --blur if and only if your model was trained that way.

Then, run
```
python generate_images.py --eval
```
And your latent vectors will be turned into their corresponding StyleGAN faces.
That's all!

### References

https://github.com/phillipi/pix2pix/blob/master/scripts/edges/

https://git.droidware.info/wchen342/SketchyGAN

https://github.com/zllrunning/face-parsing.PyTorch

https://github.com/sniklaus/pytorch-hed/blob/master/run.py
