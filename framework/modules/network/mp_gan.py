import tensorflow as tf
from ...utils.log import log_message

from ..module_base import module_base
from ..codebase.layers import normalizeInput
from ..codebase.visualization import packVoxelData

from ..codebase import net_structures

import tensorflow_probability as tfp

class voxel_generator(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        global_data_dict = kwargs['global_data_dict']

        log_message('voxel_generator', '---Building subgraph...---')

        gpu_list = cfg_args.gpu_list

        if(phase == 'train'):
            data_loader = kwargs['data_loader']
            batchsize_gpu = data_loader.public_ops['input_mask'][0].get_shape().as_list()[0]
        else:
            batchsize_gpu = 16
       
        if('net_g' not in global_data_dict):
            net_generator = getattr(net_structures, cfg_args.network.type)(0, cfg_args = cfg_args.network)
        else:
            net_generator = global_data_dict['net_g']

        #random z
        tf_z_all_gpu = []
        tf_voxel_all_gpu = []

        with tf.name_scope('z_code'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    print(cfg_args.truncation_sigma)
                    if(cfg_args.truncation_sigma < 0 or phase == 'train'):
                        log_message('voxel_generator', '---Using normal sample...---')
                        _z = tf.random_normal(shape=[batchsize_gpu, cfg_args.network.z_dim])
                    else:
                        log_message('voxel_generator', '---Using truncated sample...---')
                        dist = tfp.distributions.TruncatedNormal(loc=0, scale=1, low = -cfg_args.truncation_sigma, high = cfg_args.truncation_sigma)
                        _z = dist.sample([batchsize_gpu, cfg_args.network.z_dim])
                    tf_z_all_gpu.append(_z)
        self.public_ops['random_z'] = tf_z_all_gpu

        #voxel generation
        with tf.name_scope('generator'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    _voxel = net_generator.forward(tf_z_all_gpu[g_id], training = True)     #always using training BN even for generation
                    tf_voxel_all_gpu.append(_voxel)
        self.public_ops['voxel_generated'] = tf_voxel_all_gpu

        if('net_g' not in global_data_dict):
            global_data_dict['net_g'] = net_generator        

        #Current we must add voxels one by one, because they actually have different num of points(voxels)
        #TODO: Find a better solution?
        #Issue: Too many mesh visualization make tensorboard extremely slow, so only visualize part of them...
        if(phase != 'test'):
            for g_id in range(0, len(gpu_list)):
                for k in range(min(4, batchsize_gpu)):
                    _voxel_pos = tf.expand_dims(tf.where(tf_voxel_all_gpu[g_id][k] > 0.5), 0)
                    _voxel_pos = tf.cast(_voxel_pos, tf.float32)    #1*N*3, Z-Y-X
                    #convert to X-Y-Z order
                    _voxel_pos = tf.reverse(_voxel_pos, axis = [-1])
                    _voxel_data = packVoxelData(_voxel_pos)
                    self.add_summary(_voxel_data, phase, 'voxel', 'voxel_{}-{}'.format(g_id, k))            

class voxel_discriminator(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        data_loader = kwargs['data_loader']
        projection = kwargs['projection']

        global_data_dict = kwargs['global_data_dict']

        log_message('voxel_discrimintor', '---Building subgraph...---')

        gpu_list = cfg_args.gpu_list
        net_discriminator = getattr(net_structures, cfg_args.network.type)(0, cfg_args = cfg_args.network)
        
        batchsize_gpu = data_loader.public_ops['input_voxel'][0].get_shape().as_list()[0]

        tf_real_d_all_gpu = []
        tf_fake_d_all_gpu = []
        with tf.name_scope('discriminator'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    with tf.name_scope('real'):
                        _real_d = net_discriminator.forward(
                            normalizeInput(data_loader.public_ops['input_voxel'][g_id]), 
                            training = True)
                    with tf.name_scope('fake'):
                        _fake_d = net_discriminator.forward(
                            normalizeInput(projection.public_ops['projected_voxel'][g_id]), 
                            training = True)

                    tf_real_d_all_gpu.append(_real_d)
                    tf_fake_d_all_gpu.append(_fake_d)

        self.public_ops['real_d'] = tf_real_d_all_gpu
        self.public_ops['fake_d'] = tf_fake_d_all_gpu

        global_data_dict['net_d'] = net_discriminator    

class multi_view_image_discriminator(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        data_loader = kwargs['data_loader']
        projection = kwargs['projection']

        global_data_dict = kwargs['global_data_dict']

        log_message('multi_view_image_discriminator', '---Building subgraph...---')

        gpu_list = cfg_args.gpu_list
        net_discriminator = getattr(net_structures, cfg_args.network.type)(0, cfg_args = cfg_args.network)
        
        batchsize_gpu = data_loader.public_ops['input_mask'][0].get_shape().as_list()[0]

        tf_real_d_all_gpu = []
        tf_fake_d_all_gpu = []
        with tf.name_scope('discriminator'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    with tf.name_scope('real'):
                        _real_d = net_discriminator.forward(
                            normalizeInput(data_loader.public_ops['input_mask'][g_id]), 
                            data_loader.public_ops['input_cluster_label'][g_id], 
                            training = True)
                    with tf.name_scope('fake'):
                        _fake_d = net_discriminator.forward(
                            normalizeInput(projection.public_ops['projected_mask'][g_id]), 
                            data_loader.public_ops['input_cluster_label'][g_id], 
                            training = True)

                    tf_real_d_all_gpu.append(_real_d)
                    tf_fake_d_all_gpu.append(_fake_d)

        self.public_ops['real_d'] = tf_real_d_all_gpu
        self.public_ops['fake_d'] = tf_fake_d_all_gpu

        global_data_dict['net_d'] = net_discriminator        

class image_classifier(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        data_loader = kwargs['data_loader']

        global_data_dict = kwargs['global_data_dict']

        log_message('image_classifier', '---Building subgraph...---')

        if(phase == 'train'):
            gpu_list = cfg_args.gpu_list
            batchsize_gpu = data_loader.public_ops['input_mask'][0].get_shape().as_list()[0]
        else:
            gpu_list = [0]
            batchsize_gpu = 1

        if('net_c' not in global_data_dict):
            net_classifier = getattr(net_structures, cfg_args.network.type)(0, cfg_args = cfg_args.network)
        else:
            net_classifier = global_data_dict['net_c']

        tf_logit_all_gpu = []
        tf_prob_all_gpu = []
        with tf.name_scope('classifier_{}'.format(phase)):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    _training_tag = True if phase == 'train' else False
                    _logit = net_classifier.forward(normalizeInput(data_loader.public_ops['input_mask'][g_id]), training = _training_tag)
                    _prob = tf.nn.softmax(_logit)
                    tf_logit_all_gpu.append(_logit)
                    tf_prob_all_gpu.append(_prob)

        self.public_ops['predict_logit'] = tf_logit_all_gpu
        self.public_ops['predict_prob'] = tf_prob_all_gpu

        if('net_c' not in global_data_dict):
            global_data_dict['net_c'] = net_classifier