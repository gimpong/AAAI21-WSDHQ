import sys
import logging
import warnings

import dataset
from net import WSDHQ
from util import set_logger, DataLoader

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


### Define input arguments
ds = sys.argv[1] # dataset
lr = float(sys.argv[2])
iter_num = int(sys.argv[3])
lambda_ = float(sys.argv[4])
subspace_num = int(sys.argv[5])
loss_name = sys.argv[6]
notes = sys.argv[7]
gpu = sys.argv[8]

if ds not in ['nuswide', 'flickr']:
    print(f"unknown dataset '{ds}', use 'nus-wide' by default.")
    ds = 'nuswide'

ds2path_map = {
    'nuswide': 'nus-wide',
    'flickr': 'flickr25k'
}

ds2img_data_root_map = {
    'nuswide': './datasets/nus-wide/',
    'flickr': './datasets/flickr25k/'
}

config = {
    ## training settings
    'dataset': ds, 
    'train': True, 
    'device': '/gpu:' + gpu,
    'max_iter': iter_num,
    'batch_size': 256,
    'output_dim': 300,
    'learning_rate': lr,      # Initial learning rate img.
    'lr_decay_step': 300,     # Epochs after which learning rate decays.
    'lr_decay_factor': 0.5,   # Learning rate decay factor.

    ## dataset, tags and checkpoint I/Os
    # take tags as labels to train WSDHQ
    'img_data_root': ds2img_data_root_map[ds],
    'train_img_fpath': f"./data/{ds2path_map[ds]}/train_img.txt",
    'train_label_fpath': f"./data/{ds2path_map[ds]}/train_tag.txt",
    'idx_transform_fpath': f"./data/{ds2path_map[ds]}/tags/TagIdMergeMap.pkl",
    'final_tag_embs_fpath': f"./data/{ds2path_map[ds]}/tags/FinalTagEmbs.txt",
    'notes': notes,
    'save_ckpts_during_train': False,
    'save_ckpts_period': 200, 

    ## backbone
    'img_model': 'alexnet',
    'model_weights_fpath': './alexnet.npy',
    'finetune_all': True,    # finetune all layers, or only finetune last layer

    ## loss function
    'loss_name': loss_name,
    'use_l2_norm': True, 
    'use_neg_sampling': False,
    'use_adaptive_margin': True, 
    'margin': 0.7,
    'gamma': 1, 

    ## quantization
    'lambda': lambda_,
    'code_update_epoch_period': 2,
    'max_iter_update_b': 3,
    'max_iter_update_Cb': 1,
    'code_batch_size': 500,
    'subspace_num': subspace_num,
    'subcenter_num': 256,
}

set_logger(config)
logging.info("prepare dataset")
train_dataset = dataset.import_train(config)
logging.info("prepare data loader")
dataloader = DataLoader(
    train_dataset,
    config['output_dim'],
    config['subspace_num'] * config['subcenter_num'])
logging.info("prepare model")
model = WSDHQ(config)
logging.info("begin training")
model.train(dataloader)
logging.info("finish training, model saved under " + model.save_path + ".npy")
