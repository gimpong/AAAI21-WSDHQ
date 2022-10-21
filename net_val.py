import os
import time
import random
import logging
import h5py as h5
from tqdm import tqdm
from math import ceil

import numpy as np
import tensorflow as tf

from util import MAPs_tf, MAPs_np


class WSDHQ:
    def __init__(self, config):
        ## basic settings
        np.set_printoptions(precision=4)
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.output_dim = config['output_dim']

        ## backbone
        self.img_model = config['img_model']

        ## quantization
        self.max_iter_update_b = config['max_iter_update_b']
        self.code_batch_size = config['code_batch_size']
        self.subspace_num = config['subspace_num']
        self.subcenter_num = config['subcenter_num']

        ## tags and checkpoint I/Os
        self.wordvec_dict = np.loadtxt(config['final_tag_embs_fpath'])
        self.label_num = len(self.wordvec_dict)
        logging.info("Number of semantic embeddings: %d" % self.label_num)
        self.model_weights_fpath = config['model_weights_fpath']
        self.save_path = config['save_path']

        ## tensorflow session
        configProto = tf.compat.v1.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.compat.v1.Session(config=configProto)

        ## Create variables and placeholders
        with tf.device(self.device):
            self.images = tf.compat.v1.placeholder(  # BxImgShape
                dtype=tf.float32, 
                shape=[self.batch_size, 256, 256, 3])
            self.wordvec_dict = tf.constant(self.wordvec_dict, dtype=tf.float32) # TxD
            self.img_feats, _, self.C = self.load_model(self.model_weights_fpath)

            self.ICM_b = tf.compat.v1.placeholder( # BxMK
                dtype=tf.float32,
                shape=[None, self.subcenter_num * self.subspace_num])
            self.ICM_m = tf.compat.v1.placeholder( # scalar
                dtype=tf.int32,
                shape=[])
            begin_idx = self.ICM_m * self.subcenter_num
            end_idx = begin_idx + self.subcenter_num
            ICM_b_m = self.ICM_b[:, begin_idx: end_idx] # BxK
            ICM_C_m = self.C[begin_idx: end_idx] # KxD
            self.ICM_img_feats = tf.compat.v1.placeholder( # BxD
                dtype=tf.float32,
                shape=[self.code_batch_size, self.output_dim])
            # BxD - BxMK * MKxD + BxK * KxD => BxD
            ICM_img_feats_residual = self.ICM_img_feats - \
                tf.matmul(self.ICM_b, self.C) + \
                tf.matmul(ICM_b_m, ICM_C_m)
            ICM_cosine_similarity_quantization_residual = tf.reshape( # BKxT => BxKxT
                tf.matmul( # BKxD * DxT => BKxT
                    tf.reshape( # Bx1xD - 1xKxD => BxKxD => BKxD
                        tf.expand_dims(ICM_img_feats_residual, 1) - tf.expand_dims(ICM_C_m, 0), 
                        [self.code_batch_size * self.subcenter_num, self.output_dim]), 
                    tf.transpose(self.wordvec_dict)), 
                [self.code_batch_size, self.subcenter_num, self.label_num])
            ICM_cosine_similarity_quantization_error = tf.reduce_sum(  # BxKxT => BxK
                tf.square(ICM_cosine_similarity_quantization_residual), 
                reduction_indices=2)
            ICM_best_centers = tf.argmin(ICM_cosine_similarity_quantization_error, 1) # BxK => B
            self.ICM_best_centers_one_hot = tf.one_hot( # BxK
                indices=ICM_best_centers,
                depth=self.subcenter_num,
                dtype=tf.float32)

            self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def load_model(self, model_weights_fpath):
        if self.img_model == 'alexnet':
            img_output = self.alexnet_model(model_weights_fpath)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def alexnet_model(self, model_weights_fpath):
        model_weights = np.load(model_weights_fpath, allow_pickle=True, encoding="latin1").item()
        
        ## swap(2,1,0)
        reshaped_image = tf.cast(self.images, tf.float32)
        tm = tf.Variable([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        reshaped_image = tf.reshape(reshaped_image, [self.batch_size * 256 * 256, 3])
        reshaped_image = tf.matmul(reshaped_image, tm)
        reshaped_image = tf.reshape(reshaped_image, [self.batch_size, 256, 256, 3])
        
        IMAGE_SIZE = 227
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        distorted_image = []
        for flip_func in [tf.image.flip_left_right, lambda x: x]:
            for offset in [(0, 0), (28, 28), (28, 0), (0, 28), (14, 14)]:
                distorted_image.append( # 10x[BxImgShape]
                    tf.stack([ # BxImgShape
                        tf.image.crop_to_bounding_box( # ImgShape
                            flip_func(image),
                            *offset, height, width
                        ) for image in tf.unstack(reshaped_image)]))
        distorted_image = tf.concat(distorted_image, 0) # 10BxImgShape

        ### Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant(
                [103.939, 116.779, 123.68],
                dtype=tf.float32,
                shape=[1, 1, 1, 3],
                name='img-mean')
            distorted_image = distorted_image - mean

        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(model_weights['conv1'][0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(model_weights['conv1'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)

        ### Pool1
        self.pool1 = tf.nn.max_pool2d(self.conv1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool1')

        ### LRN1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv2
        ### Output 256, pad 2, kernel 5, group 2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(model_weights['conv2'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.lrn1, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            
            biases = tf.Variable(model_weights['conv2'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)

        ### Pool2
        self.pool2 = tf.nn.max_pool2d(self.conv2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')

        ### LRN2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv3
        ### Output 384, pad 1, kernel 3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(model_weights['conv3'][0], name='weights')
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(model_weights['conv3'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)

        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(model_weights['conv4'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.conv3, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.Variable(model_weights['conv4'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)

        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(model_weights['conv5'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(self.conv4, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            biases = tf.Variable(model_weights['conv5'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)

        ### Pool5
        self.pool5 = tf.nn.max_pool2d(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')

        ### FC6
        ### Output 4096
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(model_weights['fc6'][0], name='weights')
            fc6b = tf.Variable(model_weights['fc6'][1], name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            # self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), rate=0.5) # train, rate=1-keep_prob
            self.fc6 = tf.nn.relu(fc6l)
            self.fc6o = tf.nn.relu(fc6l)

        ### FC7
        ### Output 4096
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(model_weights['fc7'][0], name='weights')
            fc7b = tf.Variable(model_weights['fc7'][1], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            # self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), rate=0.5) # train, rate=1-keep_prob
            self.fc7 = tf.nn.relu(fc7l)
            fc7lo = tf.nn.bias_add(tf.matmul(self.fc6o, fc7w), fc7b)            
            self.fc7o = tf.nn.relu(fc7lo)

        ### FC8
        ### Output output_dim
        with tf.name_scope('fc8') as scope:
            ### Differ train and val stage by 'fc8' as key
            if 'fc8' in model_weights:
                fc8w = tf.Variable(model_weights['fc8'][0], name='weights')
                fc8b = tf.Variable(model_weights['fc8'][1], name='biases')
            else:
                fc8w = tf.Variable(tf.random.normal([4096, self.output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
                fc8b = tf.Variable(tf.constant(0.0, shape=[self.output_dim],
                                               dtype=tf.float32), name='biases')
            fc8l = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            self.fc8_t = tf.nn.tanh(fc8l) # 10BxD
            # 10BxD => 10x[BxD] => 10xBxD => BxD
            self.fc8 = tf.reduce_mean(tf.stack(tf.split(self.fc8_t, 10)), 0)
            fc8lo = tf.nn.bias_add(tf.matmul(self.fc7o, fc8w), fc8b)
            self.fc8o = tf.nn.tanh(fc8lo)

        ### load centers
        if 'C' in model_weights:
            self.centers = tf.Variable(model_weights['C'], name='centers')
        else:
            self.centers = tf.Variable(tf.random.uniform(
                shape=[self.subspace_num * self.subcenter_num, self.output_dim],
                minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))

        ### Return outputs
        return self.fc8, self.fc8o, self.centers

    def save_retrieval(self, database, query, C, retrieval_fpath=None):
        if retrieval_fpath is None:
            retrieval_fpath = self.save_path + "_retrieval.h5"
        retrieval_info = {}
        with h5.File(retrieval_fpath, 'w') as retrieval_file:
            retrieval_info['db_features'] = database.img_feats
            retrieval_file.create_dataset(
                'db_features', data=retrieval_info['db_features'])
            retrieval_info['db_reconstr'] = np.dot(database.codes, C)
            retrieval_file.create_dataset(
                'db_reconstr', data=retrieval_info['db_reconstr'])
            retrieval_info['db_label'] = database.label
            retrieval_file.create_dataset(
                'db_label', data=retrieval_info['db_label'])
            retrieval_info['qry_features'] = query.img_feats
            retrieval_file.create_dataset(
                'qry_features', data=retrieval_info['qry_features'])
            retrieval_info['qry_reconstr'] = np.dot(query.codes, C)
            retrieval_file.create_dataset(
                'qry_reconstr', data=retrieval_info['qry_reconstr'])
            retrieval_info['qry_label'] = query.label
            retrieval_file.create_dataset(
                'qry_label', data=retrieval_info['qry_label'])
        return retrieval_info

    def load_retrieval(self, retrieval_fpath=None):
        if retrieval_fpath is None:
            retrieval_fpath = self.save_path + "_retrieval.h5"
        retrieval_info = {}
        with h5.File(retrieval_fpath, "r") as retrieval_file:
            retrieval_info['db_features'] = retrieval_file['db_features'][()]
            retrieval_info['db_reconstr'] = retrieval_file['db_reconstr'][()]
            retrieval_info['db_label'] = retrieval_file['db_label'][()]
            retrieval_info['qry_features'] = retrieval_file['qry_features'][()]
            retrieval_info['qry_reconstr'] = retrieval_file['qry_reconstr'][()]
            retrieval_info['qry_label'] = retrieval_file['qry_label'][()]
        return retrieval_info

    def apply_ICM(self, img_feats, codes):
        '''
        Optimize:
            min || img_feats - self.C * codes ||
            min || img_feats - codes * self.C ||
        args:
            img_feats: [batch_size, output_dim]
            self.C: [subspace_num * subcenter_num, output_dim]
                [C_1, C_2, ... C_M]
            codes: [batch_size, subspace_num * subcenter_num]
        '''
        codes = np.zeros(codes.shape)
        for iter in range(self.max_iter_update_b):
            sub_list = list(range(self.subspace_num))
            random.shuffle(sub_list)
            for m in sub_list:
                best_centers_one_hot = self.sess.run(
                    self.ICM_best_centers_one_hot, 
                    feed_dict={
                        self.ICM_b: codes,
                        self.ICM_img_feats: img_feats,
                        self.ICM_m: m})
                codes[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot
        return codes

    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        dataset.start_epoch()
        total_batch = int(ceil(dataset.n_samples / batch_size))
        for i in tqdm(range(total_batch), desc="update batch codes"):
            img_feats, curr_codes_batch = dataset.next_batch_output_codes(batch_size)
            updated_codes_batch = self.apply_ICM(img_feats, curr_codes_batch)
            dataset.feed_batch_codes(updated_codes_batch)
        logging.info("number of update_code wrong: {}".format(
            np.sum(np.sum(dataset.codes, 1) != self.subspace_num)))
        return

    def validation(self, qry_dataloader, db_dataloader, topK=100, 
                   reload_if_exists=True, 
                   evaluator_type='tf', 
                   metric_mode='111'): # AQD, SQD, feats
        if reload_if_exists and os.path.exists(self.save_path + "_retrieval.h5"):
            logging.info("reload retrieval information: codes, features, reconstructions of queries and database")
            model_codes = self.load_retrieval()
        else:
            total_qry_batch = int(ceil(qry_dataloader.n_samples / self.batch_size))
            start_time = time.time()
            for i in tqdm(range(total_qry_batch), desc="qry feat extract"):
                images = qry_dataloader.next_batch(self.batch_size)[0]
                img_feats = self.sess.run(self.img_feats, 
                                        feed_dict={self.images: images})
                qry_dataloader.feed_batch_img_feats(img_feats)
            logging.info("finish query feature extraction, duration: %.2f sec" % (time.time() - start_time))

            total_db_batch = int(ceil(db_dataloader.n_samples / self.batch_size))
            start_time = time.time()
            for i in tqdm(range(total_db_batch), desc="db_feat extract"):
                images = db_dataloader.next_batch(self.batch_size)[0]
                img_feats = self.sess.run(self.img_feats, 
                                        feed_dict={self.images: images})
                db_dataloader.feed_batch_img_feats(img_feats)
            logging.info("finish database feature extraction, duration: %.2f sec" % (time.time() - start_time))

            logging.info("compute quantization codes for query")
            start_time = time.time()
            self.update_codes_batch(qry_dataloader, self.code_batch_size)
            logging.info("finish query encoding, duration: %.2f sec" % (time.time() - start_time))
            logging.info("compute quantization codes for database")
            start_time = time.time()
            self.update_codes_batch(db_dataloader, self.code_batch_size)
            logging.info("finish database encoding, duration: %.2f sec" % (time.time() - start_time))
            
            logging.info("save retrieval information: codes, features, reconstructions of queries and database")
            model_codes = self.save_retrieval(db_dataloader, qry_dataloader, self.sess.run(self.C))

        logging.info("begin to calculate MAP@%d" % topK)
        if evaluator_type == 'tf':
            # tensorflow version
            mAPs = MAPs_tf(self.sess, model_codes, 
                           topK, self.batch_size, self.device)
        elif evaluator_type == 'np':
            # numpy version
            mAPs = MAPs_np(model_codes, topK)
        else:
            raise NotImplementedError("evaluator_type must be chosen from 'tf' or 'np'")

        if metric_mode[0] == '1':  # AQD
            logging.info("begin to calculate AQD mAP@%d" % topK)
            start_time = time.time()
            logging.info("AQD mAP@%d = [%6.4f], duration: %.2f sec" %
                         (topK, mAPs.get_mAPs_AQD(), time.time() - start_time))
        if metric_mode[1] == '1':  # SQD
            logging.info("begin to calculate SQD mAP@%d" % topK)
            start_time = time.time()
            logging.info("SQD mAP@%d = [%6.4f], duration: %.2f sec" %
                         (topK, mAPs.get_mAPs_SQD(), time.time() - start_time))
        if metric_mode[2] == '1':  # feats
            logging.info("begin to calculate feats mAP@%d" % topK)
            start_time = time.time()
            logging.info("feats mAP@%d = [%6.4f], duration: %.2f sec" %
                         (topK, mAPs.get_mAPs_feats(), time.time() - start_time))
        return
