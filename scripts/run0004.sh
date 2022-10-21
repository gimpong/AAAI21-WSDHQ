#!/bin/bash

cd ..

##32 bits
#                     dataset  lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       flickr   0.0003  800   0.0001    4             WSDQH  0004   $1
#                     dataset  model_weight                                                                  gpu
python validation.py  flickr   ./checkpoints/flickr_WSDQH_nbits=32_adaMargin_gamma=1_lambda=0.0001_0004.npy  $1

cd -
