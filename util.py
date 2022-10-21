import os
import logging
from tqdm import tqdm
from collections import namedtuple

import numpy as np
import tensorflow as tf


def set_logger(config):
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)
    
    if config['train'] is True:
        exp_name = config['dataset']
        exp_name += '_' + config['loss_name']
        exp_name += '_nbits=' + str(config['subspace_num'] * 8)
        if config['use_adaptive_margin']:
            exp_name += '_adaMargin_gamma=' + str(config['gamma'])
        else:
            exp_name += '_fixMargin_margin=' + str(config['margin'])
        exp_name += '_lambda=' + str(config['lambda'])
        exp_name += '_' + str(config['notes'])
    else: # test
        exp_name = os.path.split(config['model_weights_fpath'])[-1][:-4]
    log_file = os.path.join('./logs/', exp_name + '.log')
    config['save_path'] = "./checkpoints/" + exp_name

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s.',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    return


class DataLoader(object):
    def __init__(self, dataset, output_dim, code_dim):
        logging.info("Initializing DataLoader")
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        self._codes = np.zeros((self.n_samples, code_dim), dtype=np.float32)

        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        logging.info("DataLoader already")
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Training stage need repeating get batch
                self._epochs_complete += 1
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch
        
        data, label = self._dataset.data(self._perm[start: end])
        return (data, label, self.codes[self._perm[start: end]])

    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch
        
        return (self.img_feats[self._perm[start: end]],
                self.codes[self._perm[start: end]])

    def feed_batch_img_feats(self, img_feats):
        """
        Args:
          [batch_size, output_dim]
        """
        start = self._index_in_epoch - len(img_feats)
        end = self._index_in_epoch
        self.img_feats[self._perm[start:end]] = img_feats
        return

    def feed_batch_codes(self, codes):
        """
        Args:
          [batch_size, MK]
        """
        start = self._index_in_epoch - len(codes)
        end = self._index_in_epoch
        self.codes[self._perm[start:end]] = codes
        return

    @property
    def img_feats(self):
        return self._output
    
    @property
    def codes(self):
        return self._codes

    @property
    def label(self):
        return self._dataset.get_labels()
    
    def start_epoch(self):
        self._index_in_epoch = 0
        np.random.shuffle(self._perm)


# tensorflow version
class MAPs_tf:
    def __init__(self, sess, retrieval_info, topK=None, batch_size=None, device='cpu'):
        self.db_features = retrieval_info['db_features']
        self.db_reconstr = retrieval_info['db_reconstr']
        self.db_label = retrieval_info['db_label']
        self.qry_features = retrieval_info['qry_features']
        self.qry_reconstr = retrieval_info['qry_reconstr']
        self.qry_label = retrieval_info['qry_label']
        self.n_db = len(self.db_features)
        self.n_qry = len(self.qry_features)
        self.output_dim = self.db_features.shape[-1]
        self.label_dim = self.db_label.shape[-1]
        self.topK = topK if topK else self.n_db
        self.batch_size = batch_size if batch_size else self.n_qry
        assert self.n_qry % self.batch_size == 0
        self.device = device
        self.sess = sess

        with tf.device(self.device):
            self.query_embs = tf.compat.v1.placeholder(tf.float32, [None, self.output_dim]) # BxD            
            self.database_embs = tf.compat.v1.placeholder(tf.float32, self.db_features.shape) # NdxD
            similarities = tf.matmul(  # similarities: BxNd
                self.query_embs, self.database_embs, transpose_b=True)
            top_rel_ids = tf.math.top_k(similarities, self.topK).indices # BxtopK
            row_ids = tf.tile( # B => Bx1 => BxtopK
                tf.expand_dims(tf.range(self.batch_size), -1),
                [1, self.topK]
            )
            top_rel_ids = tf.stack([row_ids, top_rel_ids], -1) # BxtopKx2

            self.query_label = tf.compat.v1.placeholder(tf.int32, [None, self.label_dim]) # BxL
            database_label = tf.convert_to_tensor(self.db_label, tf.int32) # NdxL
            matches = tf.cast(tf.cast(  # matches: BxNd
                tf.matmul(self.query_label, database_label, transpose_b=True), 
            tf.bool), tf.float32)
            top_matches = tf.gather_nd(matches, top_rel_ids) # BxtopK

            rel = tf.reduce_sum(top_matches, -1) # B
            rel_nonzero_flag = tf.greater(rel, 10e-6) # B
            self.rel_nonzero_flag = tf.cast(rel_nonzero_flag, rel.dtype) # B
            Lx = tf.cumsum(top_matches, -1)  # BxtopK
            position = tf.range(start=1, limit=self.topK + 1, dtype=tf.float32) # topK
            self.Px = Lx / position  # BxtopK / topK => BxtopK
            # avoid div 0
            rel = tf.where(rel_nonzero_flag, rel, tf.ones_like(rel))
            self.Rx = Lx / tf.expand_dims(rel, 1)  # BxtopK / Bx1 => BxtopK
            self.APx = tf.reduce_sum(self.Px * top_matches, -1) / rel  # B / B => B

    def _get_metrics(self, qry_embs, db_embs, notes):
        total_batch = self.n_qry // self.batch_size
        # Nbx[B], Nbx[BxtopK], Nbx[BxtopK], Nbx[B]
        all_flag, all_Px, all_Rx, all_APx = [], [], [], []
        for i in tqdm(range(total_batch), desc="compute %s mAP by batch" % notes):
            batch_flag, batch_Px, batch_Rx, batch_APx = self.sess.run(
                [self.rel_nonzero_flag, self.Px, self.Rx, self.APx],
                feed_dict={
                    self.query_embs: qry_embs[i * self.batch_size: (i + 1) * self.batch_size],
                    self.database_embs: db_embs,
                    self.query_label: self.qry_label[i * self.batch_size: (i + 1) * self.batch_size]
                }
            )
            all_flag.append(batch_flag)
            all_Px.append(batch_Px)
            all_Rx.append(batch_Rx)
            all_APx.append(batch_APx)
        flag_sum = np.sum(all_flag)
        precisions = np.sum(np.concatenate(all_Px), axis=0) / flag_sum # Nbx[BxtopK] => NbBxtopK => topK
        recalls = np.sum(np.concatenate(all_Px), axis=0) / flag_sum # Nbx[BxtopK] => NbBxtopK => topK
        mAP = np.sum(all_APx) / flag_sum
        return mAP, precisions, recalls
    
    def get_mAPs_SQD(self, RP_fpath=None):
        mAP, Px, Rx = self._get_metrics(
            qry_embs=self.qry_reconstr,
            db_embs=self.db_reconstr, 
            notes='SQD')
        if RP_fpath:
            np.savetxt(RP_fpath, np.stack([Rx, Px]))
        return mAP

    def get_mAPs_AQD(self, RP_fpath=None):
        mAP, Px, Rx = self._get_metrics(
            qry_embs=self.qry_features,
            db_embs=self.db_reconstr,
            notes='AQD')
        if RP_fpath:
            np.savetxt(RP_fpath, np.stack([Rx, Px]))
        return mAP

    def get_mAPs_feats(self, RP_fpath=None):
        mAP, Px, Rx = self._get_metrics(
            qry_embs=self.qry_features,
            db_embs=self.db_features,
            notes='feats')
        if RP_fpath:
            np.savetxt(RP_fpath, np.stack([Rx, Px]))
        return mAP


def np_topK(array, topK=1, axis=-1, largest=True, sorted=True):
    ## Reference: https://blog.csdn.net/danengbinggan33/article/details/112525700
    np_top_k_results = namedtuple('np_top_k_results', 'values indices')
    if largest:
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-topK, axis=axis),
                                  range(axis_length - topK, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=topK, axis=axis), range(0, topK), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    if sorted:
        sorted_index = np.argsort(top_scores, axis=axis)
        if largest:
            sorted_index = np.flip(sorted_index, axis=axis)
        top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
        top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
        return np_top_k_results(top_sorted_scores, top_sorted_indexes)
    else:
        return np_top_k_results(top_scores, partition_index)


# numpy version
class MAPs_np:
    def __init__(self, retrieval_info, topK=None):
        self.db_features = retrieval_info['db_features']
        self.db_reconstr = retrieval_info['db_reconstr']
        self.db_label = retrieval_info['db_label']
        self.qry_features = retrieval_info['qry_features']
        self.qry_reconstr = retrieval_info['qry_reconstr']
        self.qry_label = retrieval_info['qry_label']
        self.n_db = len(self.db_features)
        self.n_qry = len(self.qry_features)
        self.output_dim = self.db_features.shape[-1]
        self.label_dim = self.db_label.shape[-1]
        self.topK = topK if topK else self.n_db

    def _get_metrics(self, qry_embs, db_embs, notes):
        similarities = np.dot(qry_embs, db_embs.T)
        top_rel_ids = np_topK(similarities, self.topK).indices # NqxtopK
        all_Rx, all_Px, all_mAP = [], [], []
        for i in tqdm(range(similarities.shape[0]), desc="compute %s mAP" % notes):
            label = self.qry_label[i] # L
            label[label == 0] = -1
            matches = np.sum(self.db_label[top_rel_ids[i]] == label, 1) > 0
            rel = np.sum(matches)
            if rel == 0:
                continue
            Lx = np.cumsum(matches)
            Rx = Lx / rel
            Px = Lx.astype(float) / np.arange(1, self.topK + 1)
            all_Rx.append(Rx)
            all_Px.append(Px)
            all_mAP.append(np.sum(Px * matches) / rel)
        return np.mean(all_mAP), np.mean(np.stack(all_Px), 0), np.mean(np.stack(all_Rx), 0)

    def get_mAPs_SQD(self, RP_fpath=None):
        mAP, Px, Rx = self._get_metrics(
            qry_embs=self.qry_reconstr,
            db_embs=self.db_reconstr,
            notes='SQD')
        if RP_fpath:
            np.savetxt(RP_fpath, np.stack([Rx, Px]))
        return mAP

    def get_mAPs_AQD(self, RP_fpath=None):
        mAP, Px, Rx = self._get_metrics(
            qry_embs=self.qry_features,
            db_embs=self.db_reconstr,
            notes='AQD')
        if RP_fpath:
            np.savetxt(RP_fpath, np.stack([Rx, Px]))
        return mAP

    def get_mAPs_feats(self, RP_fpath=None):
        mAP, Px, Rx = self._get_metrics(
            qry_embs=self.qry_features,
            db_embs=self.db_features,
            notes='feats')
        if RP_fpath:
            np.savetxt(RP_fpath, np.stack([Rx, Px]))
        return mAP

