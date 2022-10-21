# =============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the NUS-WIDE binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import pickle as pk
import cv2
import numpy as np

# Process images of this size. Note that this differs from the original nus-wide
# image size of 224 x 224. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

class Dataset:
    def __init__(self, img_path, lb_path, 
                 img_prefix='./', 
                 idx_transform_fpath=None, train=True):
        self.img_lines = open(img_path, 'r').readlines()
        self.img_prefix = img_prefix
        self.n_samples = len(self.img_lines)
        self.train = train
        self._img  = [0] * self.n_samples
        self._label = np.loadtxt(lb_path)
        if self.train:
            self.idx_transform_fpath = idx_transform_fpath
            self.merge_labels()
        self._n_label = self._label.shape[1]
        self._load = [0] * self.n_samples
        self._load_num = 0            
        self._status = 0
        self.data = self.img_data
        self.all_data = self.img_all_data

    def img_data(self, index):
        if self._status:
            return (self._img[index, :], self._label[index, :])
        else:
            ret_img = []
            for i in index:
                try:
                    if self.train:
                        if not self._load[i]:
                            self._img[i] = cv2.resize(
                                cv2.imread(os.path.join(
                                    self.img_prefix, 
                                    self.img_lines[i].strip())), (256, 256))
                            self._load[i] = 1
                            self._load_num += 1
                        ret_img.append(self._img[i])
                    else:
                        ret_img.append(cv2.resize(
                            cv2.imread(os.path.join(
                                self.img_prefix, 
                                self.img_lines[i].strip())), (256, 256)))
                except:
                    logging.info('cannot open' + self.img_lines[i])
                #else:
                    #logging.info(self.img_lines[i])
                
            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
            return (np.asarray(ret_img), self._label[index, :])

    def img_all_data(self):
        if self._status:
            return (self._img, self._label)

    def get_labels(self):
        return self._label
    
    def merge_labels(self):
        label_cols = self._label.T
        idx_transform = pk.load(open(self.idx_transform_fpath, 'rb'))
        if idx_transform.get(-1):
            del idx_transform[-1]
        final_label_cols = []
        for label, idx_list in idx_transform.items():
            final_label_cols.append(label_cols[idx_list].any(0))
        self._label = np.stack(final_label_cols).T

    @property
    def n_label(self):
        return self._n_label


def import_train(config):
    '''
    return (train_img_fpath, txt_tr)
    '''
    return (Dataset(config["train_img_fpath"], 
                    config["train_label_fpath"],
                    img_prefix=config['img_data_root'], 
                    idx_transform_fpath=config['idx_transform_fpath'],
                    train=True))


def import_validation(config):
    '''
    return (test_img_fpath, txt_te, database_img_fpath, txt_db)
    '''
    return (Dataset(config["test_img_fpath"],
                    config["test_label_fpath"],
                    img_prefix=config['img_data_root'],
                    train=False),
            Dataset(config["database_img_fpath"],
                    config["database_label_fpath"],
                    img_prefix=config['img_data_root'],
                    train=False))
