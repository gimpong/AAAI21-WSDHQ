#!/bin/bash

cd ..

##24 bits
#                     dataset   lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       nuswide   0.0003  1500  0.0001    3             WSDQH  0007   $1
#                     dataset   model_weight                                                                   gpu
python validation.py  nuswide   ./checkpoints/nuswide_WSDQH_nbits=24_adaMargin_gamma=1_lambda=0.0001_0007.npy  $1

cd -
