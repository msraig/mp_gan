import tensorflow as tf
from ...utils.log import log_message

from ..codebase.loss_func import *
from ..codebase.layers import interpolate_for_gp, normalizeInput

from ..module_base import module_base


class loss_gan(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        data_loader = kwargs['data_loader']
        generator = kwargs['generator']
        projection = kwargs['projection']
        discriminator = kwargs['discriminator']
        global_data_dict = kwargs['global_data_dict']

        log_message('loss_gan', '---Building subgraph...---')
        gpu_list = cfg_args.gpu_list

        tf_d_loss_all_gpu = []
        tf_g_loss_all_gpu = []
        tf_gp_loss_all_gpu = []
        tf_tv_voxel_loss_all_gpu = []
        tf_beta_prior_loss_all_gpu = []
        
        with tf.name_scope('loss'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    with tf.name_scope('g_loss'):
                        if(cfg_args.gan_loss == 'hinge' or cfg_args.gan_loss == 'wgan'):
                            _loss_g = loss_wgan_g(discriminator.public_ops['fake_d'][g_id])
                        else:
                            raise NotImplementedError
                        tf_g_loss_all_gpu.append(_loss_g)

                    with tf.name_scope('d_loss'):
                        if(cfg_args.gan_loss == 'hinge'):
                            _loss_d = loss_hinge_d(discriminator.public_ops['real_d'][g_id], discriminator.public_ops['fake_d'][g_id])
                        elif(cfg_args.gan_loss == 'wgan'):
                            _loss_d = loss_wgan_d(discriminator.public_ops['real_d'][g_id], discriminator.public_ops['fake_d'][g_id])
                        else:
                            raise NotImplementedError
                        tf_d_loss_all_gpu.append(_loss_d)

                    if(cfg_args.gp > 0):
                        with tf.name_scope('gp_loss'):
                            net_d = global_data_dict['net_d']
                            _interpolate_data = interpolate_for_gp(data_loader.public_ops['input_mask'][g_id], projection.public_ops['projected_mask'][g_id])
                            _inter_d = net_d.forward(
                                normalizeInput(_interpolate_data),
                                data_loader.public_ops['input_cluster_label'][g_id],
                                training = True)
                            _loss_gp = cfg_args.gp * loss_gp(_interpolate_data, _inter_d)
                            tf_gp_loss_all_gpu.append(_loss_gp)

                    if(cfg_args.tv_voxel > 0):
                        with tf.name_scope('tv_voxel_loss'):
                            _loss_tv_voxel = cfg_args.tv_voxel * loss_tv_voxel(generator.public_ops['voxel_generated'][g_id], log_transform = True, method = 'mean')
                            tf_tv_voxel_loss_all_gpu.append(_loss_tv_voxel)                        

                    if(cfg_args.beta_prior > 0):
                        with tf.name_scope('beta_prior_loss'):
                            _loss_beta_prior = cfg_args.beta_prior * loss_beta_prior(projection.public_ops['projected_mask'][g_id])
                            tf_beta_prior_loss_all_gpu.append(_loss_beta_prior)      

        self.public_ops['loss_g'] = tf_g_loss_all_gpu
        self.add_summary(tf.reduce_mean(tf_g_loss_all_gpu), 'train', 'scalar', 'loss_g')

        self.public_ops['loss_d'] = tf_d_loss_all_gpu
        self.add_summary(tf.reduce_mean(tf_d_loss_all_gpu), 'train', 'scalar', 'loss_d')

        if(cfg_args.gp > 0):
            self.public_ops['loss_gp'] = tf_gp_loss_all_gpu
            self.add_summary(tf.reduce_mean(tf_gp_loss_all_gpu), 'train', 'scalar', 'loss_gp')

        if(cfg_args.tv_voxel > 0):
            self.public_ops['loss_tv_voxel'] = tf_tv_voxel_loss_all_gpu
            self.add_summary(tf.reduce_mean(tf_tv_voxel_loss_all_gpu), 'train', 'scalar', 'loss_tv_voxel')

        if(cfg_args.beta_prior > 0):
            self.public_ops['loss_beta_prior'] = tf_beta_prior_loss_all_gpu
            self.add_summary(tf.reduce_mean(tf_beta_prior_loss_all_gpu), 'train', 'scalar', 'loss_beta_prior')


class loss_classifier(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        data_loader = kwargs['data_loader']
        classifier = kwargs['classifier']
        
        global_data_dict = kwargs['global_data_dict']

        log_message('loss_gan', '---Building subgraph...---')

        if(phase == 'train'):
            gpu_list = cfg_args.gpu_list
        else:
            gpu_list = [0]

        tf_cross_entropy_loss_all_gpu = []

        with tf.name_scope('loss_{}'.format(phase)):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    with tf.name_scope('cross_entropy'):
                        n_class = classifier.public_ops['predict_logit'][g_id].get_shape().as_list()[-1]
                        _one_hot = tf.one_hot(data_loader.public_ops['input_view_label'][g_id], depth = n_class)
                        _loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = _one_hot, logits = classifier.public_ops['predict_logit'][g_id]))   
                        tf_cross_entropy_loss_all_gpu.append(_loss_ce)

        self.public_ops['loss_full'] = tf_cross_entropy_loss_all_gpu
        if(phase == 'train'):
            self.add_summary(tf.reduce_mean(tf_cross_entropy_loss_all_gpu), phase, 'scalar', 'loss_full')

        if(phase == 'val'):
            with tf.variable_scope('val_acc'):
                _acc, _acc_op = tf.metrics.accuracy(data_loader.public_ops['input_view_label'][0], tf.argmax(classifier.public_ops['predict_logit'][0], -1))
                stream_vars_valid = [v for v in tf.local_variables() if 'val_acc' in v.name]
                _reset_op = tf.variables_initializer(stream_vars_valid)
            
            self.add_summary(_acc, phase, 'scalar', 'val_acc')
            self.public_ops['valid_acc'] = _acc
            self.public_ops['valid_acc_op'] = _acc_op
            self.public_ops['reset_acc'] = _reset_op




        
