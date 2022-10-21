#!/bin/bash

cd ..

##32 bits
#                     dataset   lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       nuswide   0.0003  1500  0.0001    4             WSDQH  0008   $1
#                     dataset   model_weight                                                                   gpu
python validation.py  nuswide   ./checkpoints/nuswide_WSDQH_nbits=32_adaMargin_gamma=1_lambda=0.0001_0008.npy  $1

cd -
