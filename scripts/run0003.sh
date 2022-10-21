#!/bin/bash

cd ..

##24 bits
#                     dataset  lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       flickr   0.0003  800   0.0001    3             WSDQH  0003   $1
#                     dataset  model_weight                                                                  gpu
python validation.py  flickr   ./checkpoints/flickr_WSDQH_nbits=24_adaMargin_gamma=1_lambda=0.0001_0003.npy  $1

cd -
