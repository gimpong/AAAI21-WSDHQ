#!/bin/bash

cd ..

##8 bits
#                     dataset   lr      iter  lambda    subspace_num  loss   notes  gpu
python train.py       nuswide   0.0003  1500  0.0001    1             WSDQH  0005   $1
#                     dataset   model_weight                                                                  gpu
python validation.py  nuswide   ./checkpoints/nuswide_WSDQH_nbits=8_adaMargin_gamma=1_lambda=0.0001_0005.npy  $1

cd -
