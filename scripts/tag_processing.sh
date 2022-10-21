#!/bin/bash

cd ../data/flickr25k
mkdir -p tags
cd -

cd ../data/nus-wide
mkdir -p tags
cd -

cd ../data
#                      dataset  w2v_fpath                                                         gpu
python tag_preproc.py  flickr   ../datasets/GoogleNews-vectors-negative300.bin.gz  $1
#                      dataset  w2v_fpath                                                         gpu
python tag_preproc.py  nuswide  ../datasets/GoogleNews-vectors-negative300.bin.gz  $1
cd -