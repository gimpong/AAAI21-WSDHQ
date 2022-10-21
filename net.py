import time
import random
import logging
from tqdm import tqdm
from math import ceil

import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans


class WSDHQ:
    def __init__(self, config):
        ## basic settings
        np.set_printoptions(precision=4)
        self.device = config['device']
        self.max_iter = config['max_iter']
        self.batch_size = config['batch_size']
        self.output_dim = config['output_dim']
        self.learning_rate = config['learning_rate']
        self.lr_decay_step = config['lr_decay_step']
        self.lr_decay_factor = config['lr_decay_factor']

        ## backbone
        self.img_model = config['img_model']
        self.finetune_all = config['finetune_all']

        ## loss function
        self.loss_name = config['loss_name']
        self.use_l2_norm = config['use_l2_norm']
        self.use_neg_sampling = config['use_neg_sampling']
        self.use_adaptive_margin = config['use_adaptive_margin']
        # if not use adaptive margin, then fix margin as a scalar
        self.margin = config['margin']
        self.gamma = config['gamma']

        ## quantization
        self.lambda_ = config['lambda']
        self.code_update_epoch_period = config['code_update_epoch_period']
        self.max_iter_update_b = config['max_iter_update_b']
        self.max_iter_update_Cb = config['max_iter_update_Cb']
        self.code_batch_size = config['code_batch_size']
        self.subspace_num = config['subspace_num']
        self.subcenter_num = config['subcenter_num']

        ## tags and checkpoint I/Os
        self.wordvec_dict = np.loadtxt(config['final_tag_embs_fpath'])
        self.label_num = len(self.wordvec_dict)
        logging.info("Number of semantic embeddings: %d" % self.label_num)
        self.save_path = config['save_path']
        self.save_ckpts_during_train = config['save_ckpts_during_train']
        self.save_ckpts_period = config['save_ckpts_period']

        ## tensorflow session
        configProto = tf.compat.v1.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.compat.v1.Session(config=configProto)

        ## Create variables and placeholders
        with tf.device(self.device):
            self.images = tf.compat.v1.placeholder( # BxImgShape
                dtype=tf.float32, 
                shape=[self.batch_size, 256, 256, 3])
            self.labels = tf.compat.v1.placeholder(  # BxT(number of tags)
                dtype=tf.float32, 
                shape=[self.batch_size, self.label_num])
            self.wordvec_dict = tf.constant(self.wordvec_dict, dtype=tf.float32) # TxD
            self.img_feats, _, self.C = self.load_model(config['model_weights_fpath'])
            # the image embedding matrix of all images (i.e., the symbol 'R' in the paper)
            self.img_feats_all = tf.compat.v1.placeholder( # NxD
                dtype=tf.float32, 
                shape=[None, self.output_dim])
            self.img_b = tf.compat.v1.placeholder( # BxMK
                dtype=tf.float32, 
                shape=[None, self.subspace_num * self.subcenter_num])
            self.img_b_all = tf.compat.v1.placeholder( # NxMK
                dtype=tf.float32, 
                shape=[None, self.subspace_num * self.subcenter_num])

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

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def load_model(self, model_weights_fpath):
        if self.img_model == 'alexnet':
            return self.alexnet_model(model_weights_fpath)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)

    def alexnet_model(self, model_weights_fpath):
        self.model_params = {}
        self.trainable_params_group1 = []
        self.trainable_params_group2 = []
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

        ### Randomly crop a [height, width] section of each image
        distorted_image = tf.stack([
            tf.image.random_crop(
                tf.image.random_flip_left_right(each_image), 
                [height, width, 3]
            ) for each_image in tf.unstack(reshaped_image)])

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
            self.model_params['conv1'] = [kernel, biases]
            self.trainable_params_group1 += [kernel, biases]

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
            # input_groups = tf.split(3, group, self.lrn1) # tf version <= 0.12.0
            # see https://blog.csdn.net/wang2008start/article/details/71516198
            input_groups = tf.split(self.lrn1, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(output_groups, 3)
            
            biases = tf.Variable(model_weights['conv2'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.model_params['conv2'] = [kernel, biases]
            self.trainable_params_group1 += [kernel, biases]

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
            self.model_params['conv3'] = [kernel, biases]
            self.trainable_params_group1 += [kernel, biases]

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
            self.model_params['conv4'] = [kernel, biases]
            self.trainable_params_group1 += [kernel, biases]

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
            self.model_params['conv5'] = [kernel, biases]
            self.trainable_params_group1 += [kernel, biases]

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
            self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), rate=0.5)  # rate = 1 - keep_prob
            self.fc6o = tf.nn.relu(fc6l)
            self.model_params['fc6'] = [fc6w, fc6b]
            self.trainable_params_group1 += [fc6w, fc6b]

        ### FC7
        ### Output 4096
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(model_weights['fc7'][0], name='weights')
            fc7b = tf.Variable(model_weights['fc7'][1], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), rate=0.5)  # rate = 1 - keep_prob
            fc7lo = tf.nn.bias_add(tf.matmul(self.fc6o, fc7w), fc7b)            
            self.fc7o = tf.nn.relu(fc7lo)
            self.model_params['fc7'] = [fc7w, fc7b]
            self.trainable_params_group1 += [fc7w, fc7b]

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
            self.fc8 = tf.nn.tanh(fc8l)
            fc8lo = tf.nn.bias_add(tf.matmul(self.fc7o, fc8w), fc8b)
            self.fc8o = tf.nn.tanh(fc8lo)
            self.model_params['fc8'] = [fc8w, fc8b]
            self.trainable_params_group2 += [fc8w, fc8b]
        
        ### load centers
        if 'C' in model_weights:
            self.centers = tf.Variable(model_weights['C'], name='centers')
        else:
            self.centers = tf.Variable(tf.random.uniform(
                shape=[self.subspace_num * self.subcenter_num, self.output_dim],
                minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))

        self.model_params['C'] = self.centers

        ### Return outputs
        return self.fc8, self.fc8o, self.centers

    def save_model(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        model = {}
        for layer in self.model_params:
            model[layer] = self.sess.run(self.model_params[layer])
        np.save(save_path, np.array(model))
        return

    def apply_loss_function(self, global_step):
        # because tag word2vec embedding is in 300-d format
        assert self.output_dim == 300
        ### loss function
        if self.loss_name == 'WDHT':
            if self.use_l2_norm:
                img_feats = tf.nn.l2_normalize(self.img_feats, axis=-1)
                wordvec_dict = tf.nn.l2_normalize(self.wordvec_dict, axis=-1)
            else:
                img_feats = self.img_feats
                wordvec_dict = self.wordvec_dict   

            margin = tf.constant(self.margin, dtype=tf.float32)
            pos_embs = tf.expand_dims(self.labels, -1) * tf.expand_dims(wordvec_dict, 0) # BxTx1 * 1xTxD => BxTxD
            img_label_cnt = tf.reduce_sum(self.labels, -1, keepdims=True) # BxT => Bx1
            avg_pos_embs = tf.reduce_sum(pos_embs, -2) / tf.where(tf.greater(img_label_cnt, 0), img_label_cnt, tf.ones_like(img_label_cnt)) # avoid and ignore some imgs without tags
            pos_cos = tf.reduce_sum(img_feats * avg_pos_embs, -1, keepdims=True) # Bx1
            all_cos = tf.matmul(img_feats, avg_pos_embs, transpose_b=True) # BxB
            margin_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, margin + all_cos - pos_cos), -1) - margin)
            assert self.batch_size > 1
            self.embedding_loss = margin_loss / (self.batch_size - 1)

        elif self.loss_name == 'WSDQH':
            if self.use_l2_norm:
                img_feats = tf.nn.l2_normalize(self.img_feats, axis=-1)
                wordvec_dict = tf.nn.l2_normalize(self.wordvec_dict, axis=-1)
            else:
                img_feats = self.img_feats
                wordvec_dict = self.wordvec_dict    

            if self.use_adaptive_margin:
                margin = (2 ** (1 - self.gamma)) * \
                    ((1.0 - tf.matmul(wordvec_dict, wordvec_dict, transpose_b=True)) ** self.gamma) # TxT
            else: # use_adaptive_margin == False
                margin = tf.constant(self.margin, dtype=tf.float32)

            pos_embs = tf.expand_dims(self.labels, -1) * tf.expand_dims(wordvec_dict, 0) # BxTx1 * 1xTxD => BxTxD
            pos_cos = tf.reduce_sum(tf.expand_dims(img_feats, 1) * pos_embs, -1) # Bx1xD * BxTxD => BxTxD => BxT

            if self.use_neg_sampling:
                # BxTx1 * 1xTxD => BxTxD
                neg_embs = tf.expand_dims(1.0  - self.labels, -1) * tf.expand_dims(wordvec_dict, 0) 
                # Bx1xD * BxTxD => BxTxD => BxT
                neg_cos = tf.reduce_sum(tf.expand_dims(img_feats, 1) * neg_embs, -1)
                # BxT => B => ()
                max_pos_tag_cnt = tf.cast(tf.reduce_max(tf.reduce_sum(self.labels, -1)), dtype=tf.int32)
                neg_sampl_cnt = tf.minimum(self.label_num - max_pos_tag_cnt, max_pos_tag_cnt)
                neg_sampl_cos, neg_sampl_idx = tf.nn.top_k(neg_cos, neg_sampl_cnt) # BxTneg
                if self.use_adaptive_margin:
                    # gather(TxT, BxTneg) => BxTnegxT => BxTxTneg
                    margin = tf.transpose(tf.gather(margin, neg_sampl_idx), [0,2,1])
                margin_loss = tf.reduce_sum( # BxTxTneg => ()
                    tf.maximum(0.0,  # (BxTxTneg - (BxTx1 - Bx1xTneg)) * BxTx1 => BxTxTneg
                               (margin - (tf.expand_dims(pos_cos, -1) - tf.expand_dims(neg_sampl_cos, -2))) * \
                                   tf.expand_dims(self.labels, -1)))
                self.embedding_loss = margin_loss / \
                    (tf.reduce_sum(self.labels) * tf.cast(neg_sampl_cnt, tf.float32))
            else: # self.use_neg_sampling == False
                all_cos = tf.matmul(img_feats, wordvec_dict, transpose_b=True) # BxD * DxT => BxT
                # the rightest component is the selection mask to pick out the concerned elements
                margin_loss = tf.reduce_sum( # BxTxT => ()
                    tf.maximum(0.0, # (TxT - (Bx1xT - BxTx1) * Bx1xT * BxTx1) => BxTxT
                               (margin - (tf.expand_dims(pos_cos, -1) - tf.expand_dims(all_cos, -2))) * \
                                    tf.expand_dims(self.labels, -1) * \
                                    tf.expand_dims(1.0 - self.labels, -2)))
                pos_tag_cnt = tf.reduce_sum(self.labels, -1)
                neg_tag_cnt = self.label_num - pos_tag_cnt
                self.embedding_loss = margin_loss / tf.reduce_sum(pos_tag_cnt * neg_tag_cnt)
        else:
            raise NotImplementedError("Sorry, the loss '%s' is not implemented." % self.loss_name)

        self.quantization_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(tf.matmul(self.img_feats - tf.matmul(self.img_b, self.C), 
                                self.wordvec_dict, transpose_b=True)), 1))
        self.quant_loss_weight = tf.Variable(self.lambda_, name='lambda_')
        self.total_loss = self.embedding_loss + self.quant_loss_weight * self.quantization_loss

        ## optimization
        self.lr = tf.compat.v1.train.exponential_decay(self.learning_rate, global_step, self.lr_decay_step, self.lr_decay_factor, staircase=True)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        # opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.total_loss, self.trainable_params_group1+self.trainable_params_group2)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]
        ### Last layer (trainable_params_group2) has a 10 times learning rate
        if self.finetune_all:
            return opt.apply_gradients([(grads_and_vars[0][0], self.trainable_params_group1[0]),
                                        (grads_and_vars[1][0]*2, self.trainable_params_group1[1]),
                                        (grads_and_vars[2][0], self.trainable_params_group1[2]),
                                        (grads_and_vars[3][0]*2, self.trainable_params_group1[3]),
                                        (grads_and_vars[4][0], self.trainable_params_group1[4]),
                                        (grads_and_vars[5][0]*2, self.trainable_params_group1[5]),
                                        (grads_and_vars[6][0], self.trainable_params_group1[6]),
                                        (grads_and_vars[7][0]*2, self.trainable_params_group1[7]),
                                        (grads_and_vars[8][0], self.trainable_params_group1[8]),
                                        (grads_and_vars[9][0]*2, self.trainable_params_group1[9]),
                                        (grads_and_vars[10][0], self.trainable_params_group1[10]),
                                        (grads_and_vars[11][0]*2, self.trainable_params_group1[11]),
                                        (grads_and_vars[12][0], self.trainable_params_group1[12]),
                                        (grads_and_vars[13][0]*2, self.trainable_params_group1[13]),
                                        (fcgrad*10, self.trainable_params_group2[0]),
                                        (fbgrad*20, self.trainable_params_group2[1])], global_step=global_step)
        else:
            return opt.apply_gradients([(fcgrad*10, self.trainable_params_group2[0]),
                                        (fbgrad*20, self.trainable_params_group2[1])], global_step=global_step)

    def initial_centers(self, img_feats):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, self.output_dim])
        subspace_dim = self.output_dim // self.subspace_num
        for i in tqdm(range(self.subspace_num), desc="centers init"):
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(
                img_feats[:, i * subspace_dim: (i + 1) * subspace_dim])
            C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, 
                   i * subspace_dim: (i + 1) * subspace_dim] = kmeans.cluster_centers_
        return C_init

    def update_centers(self, dataloader):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        curr_C = self.sess.run(self.C)
        h = self.img_b_all
        U = self.img_feats_all
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.matmul(tf.transpose(h), h) + tf.constant(
            np.eye(self.subcenter_num * self.subspace_num, dtype=np.float32) * 0.001)
        computed_centers = tf.matmul(tf.linalg.inv(hh), Uh)
        updated_C = self.sess.run(
            self.C.assign(computed_centers), 
            feed_dict={
                self.img_feats_all: dataloader.img_feats, 
                self.img_b_all: dataloader.codes,})
        C_zeros_ids = np.where(np.sum(np.square(updated_C), axis=1) < 1e-8)
        updated_C[C_zeros_ids, :] = curr_C[C_zeros_ids, :]
        self.sess.run(self.C.assign(updated_C))
        logging.info("non zero codewords: %d" % len(np.where(np.sum(updated_C, 1) != 0)[0]))

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

    def train(self, dataloader):
        is_C_initialized = False
        total_batch = int(ceil(dataloader.n_samples / self.batch_size))
        for train_iter in range(self.max_iter):
            images, labels, codes = dataloader.next_batch(self.batch_size)
            if is_C_initialized:
                self.sess.run(self.quant_loss_weight.assign(self.lambda_))
            else:
                self.sess.run(self.quant_loss_weight.assign(0.0))

            ## get features and compute the loss
            start_time = time.time()
            _, embedding_loss, quantization_loss, quant_loss_weight, lr, img_feats = self.sess.run(
                [self.train_op, self.embedding_loss, 
                 self.quantization_loss, self.quant_loss_weight, 
                 self.lr, self.img_feats],
                feed_dict={
                    self.images: images,
                    self.labels: labels,
                    self.img_b: codes})
            dataloader.feed_batch_img_feats(img_feats)
            duration = time.time() - start_time

            logging.info("step [%4d], lr [%.7f], embedding loss [%7.4f], quantization loss [%7.4f], %5.2f sec/batch" %
                         (train_iter+1, lr, embedding_loss, quantization_loss * quant_loss_weight, duration))

            ## update codes and centers every period
            if train_iter % (self.code_update_epoch_period * total_batch) == 0 and train_iter != 0:
                if not is_C_initialized:
                    start_time = time.time()
                    with tf.device(self.device):
                        for i in range(self.max_iter_update_Cb):
                            logging.info("initialize centers iter(%d/%d)" %
                                         (i + 1, self.max_iter_update_Cb))
                            self.sess.run(self.C.assign(
                                self.initial_centers(dataloader.img_feats)))
                    logging.info("finish center initialization, duration: %.2f sec" % (time.time() - start_time))
                    is_C_initialized = True
                start_time = time.time()
                for i in range(self.max_iter_update_Cb):
                    logging.info("update codes and centers iter(%d/%d)" %
                                 (i + 1, self.max_iter_update_Cb))
                    self.update_codes_batch(dataloader, self.code_batch_size)
                    self.update_centers(dataloader)
                logging.info("finish center update, duration: %.2f sec" % (time.time() - start_time))
            
            if self.save_ckpts_during_train and train_iter != 0 and train_iter % self.save_ckpts_period == 0:
                logging.info("save checkpoint at the %d iteration")
                self.save_model(self.save_path + '_iter=%d' % train_iter)

        logging.info("finish training iterations and begin saving model")
        self.save_model()
        logging.info("finish model saving")
