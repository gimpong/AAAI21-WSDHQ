#!/bin/bash

cd ..

##8 bits
#                     dataset  lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       flickr   0.0003  800   0.0001    1             WSDQH  0001   $1
#                     dataset  model_weight                                                                 gpu
python validation.py  flickr   ./checkpoints/flickr_WSDQH_nbits=8_adaMargin_gamma=1_lambda=0.0001_0001.npy  $1

cd -
