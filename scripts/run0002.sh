#!/bin/bash

cd ..

##16 bits
#                     dataset  lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       flickr   0.0003  800   0.0001    2             WSDQH  0002   $1
#                     dataset  model_weight                                                                  gpu
python validation.py  flickr   ./checkpoints/flickr_WSDQH_nbits=16_adaMargin_gamma=1_lambda=0.0001_0002.npy  $1

cd -
