#!/bin/bash

cd ..

##16 bits
#                     dataset   lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       nuswide   0.0003  1500  0.0001    2             WSDQH  0006   $1
#                     dataset   model_weight                                                                   gpu
python validation.py  nuswide   ./checkpoints/nuswide_WSDQH_nbits=16_adaMargin_gamma=1_lambda=0.0001_0006.npy  $1

cd -
