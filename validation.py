import sys
import logging
import warnings

import dataset
from net_val import WSDHQ
from util import set_logger, DataLoader

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


### Define input arguments
ds = sys.argv[1]  # dataset
model_weight = sys.argv[2]
gpu = sys.argv[3]
subspace_num = int(model_weight.split('_')[2].split('=')[-1]) // 8

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
    ## basic settings
    'dataset': ds,
    'train': False, 
    'device': '/gpu:' + gpu,
    'batch_size': 100,
    'output_dim': 300,
    'topK': 5000,  # mAP@'topK'

    # ## dataset, tags and checkpoint I/Os
    'img_data_root': ds2img_data_root_map[ds],
    'test_img_fpath': f"./data/{ds2path_map[ds]}/test_img.txt",
    'test_label_fpath': f"./data/{ds2path_map[ds]}/test_label.txt",
    'database_img_fpath': f"./data/{ds2path_map[ds]}/database_img.txt",
    'database_label_fpath': f"./data/{ds2path_map[ds]}/database_label.txt",
    'final_tag_embs_fpath': f"./data/{ds2path_map[ds]}/tags/FinalTagEmbs.txt",

    ## backbone
    'img_model': 'alexnet',
    'model_weights_fpath': model_weight,

    ## quantization
    'max_iter_update_b': 3,
    'code_batch_size': 50 * 14,
    'subspace_num': subspace_num,
    'subcenter_num': 256, 

    ## evaluator
    'reload_if_exists': True, # if true, then use the cached code file (if exists) for evaluation
    'evaluator_type': 'np', # choose 'np' or 'tf'
    'metric_mode': '111', # AQD, SQD, feats; '1': enabled, '0': disable
}

set_logger(config)
logging.info("prepare dataset")
qry_dataset, db_dataset = dataset.import_validation(config)
logging.info("prepare data loader")
db_dataloader = DataLoader(
    db_dataset, 
    config['output_dim'], 
    config['subspace_num'] * config['subcenter_num'])
qry_dataloader = DataLoader(
    qry_dataset, 
    config['output_dim'], 
    config['subspace_num'] * config['subcenter_num'])
logging.info("prepare model")
model = WSDHQ(config)
logging.info("begin validation")
model.validation(
    qry_dataloader, db_dataloader, config['topK'],
    config['reload_if_exists'],
    config['evaluator_type'],
    config['metric_mode'])
logging.info("finish validation")
