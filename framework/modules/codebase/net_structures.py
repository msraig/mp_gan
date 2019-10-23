import tensorflow as tf
from .layers import *

#MP-GAN
class voxel_generator:
    def __init__(self, id = 0, **kwargs): #nc_f = 512, nd_f = 4, out_dim = 32, sn = 0, norm = 'bn', act = 'relu',  voxel_order = 'NDHWC', mirror = False):
        self.voxel_order_map = dict(NDHWC = 3)

        self.g_id = id
        self.inited = False
        
        cfg = kwargs['cfg_args']

        self.nc_f = cfg.nc_f
        self.nd_f = cfg.nd_f
        self.out_dim = cfg.out_dim
        self.sn = cfg.sn 
        self.norm = cfg.norm
        self.act = cfg.act
        self.voxel_order = cfg.voxel_order
        self.mirror = cfg.mirror

    def forward(self, _input, training = True):
        with tf.variable_scope('gen_{}'.format(self.g_id), reuse = self.inited):
            if(self.mirror):
                _last_dim = self.nd_f // 2
            else:
                _last_dim = self.nd_f

            with tf.variable_scope('stage_linear'):
                output = linear_sn(_input, self.nc_f * self.nd_f * self.nd_f *_last_dim, name='linear', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                output = tf.reshape(output, [-1, self.nd_f, self.nd_f, _last_dim, self.nc_f], name='reshape')
                output = normalization(output, self.norm, training = training)
                output = activation(output, self.act)

            _curr_dim = self.nd_f
            _curr_ch = self.nc_f
            while(_curr_dim < self.out_dim):
                with tf.variable_scope('stage_conv_{}-{}'.format(_curr_dim, _curr_dim * 2)):
                    output = deconv3d_sn(output, _curr_ch // 2, 4, 4, 4, 2, 2, 2, 'deconv', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                    output = normalization(output, self.norm, training = training)
                    output = activation(output, self.act)
                _curr_ch = _curr_ch // 2
                _curr_dim = _curr_dim * 2

            #final
            with tf.variable_scope('stage_conv_{}-{}'.format(_curr_dim, _curr_dim)):
                output = conv3d_sn(output, 1, 3, 3, 3, 1, 1, 1, 'conv', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                output = tf.tanh(output)
                output = denormalizeInput(output)

            if(self.mirror):
                with tf.variable_scope('stage_mirror'):
                    mirror_dim = self.voxel_order_map[self.voxel_order]
                    output = tf.concat([output, tf.reverse(output, axis = [mirror_dim])], axis = mirror_dim)

        self.inited = True
        return output

class multi_view_image_discriminator:
    def __init__(self, id = 0, **kwargs): #nc_f = 32, nd_l = 4, n_shared = 3, n_head = 8, sn = 1, norm = 'None', act = 'relu':
        self.d_id = id
        self.inited = False

        cfg = kwargs['cfg_args']

        self.nc_f = cfg.nc_f
        self.nd_l = cfg.nd_l
        self.n_head = cfg.n_head
        self.sn = cfg.sn
        self.norm = cfg.norm
        self.act = cfg.act
        self.n_shared = cfg.n_shared

    def forward(self, _image, _view, training = True):
        with tf.variable_scope('dis_{}'.format(self.d_id), reuse = self.inited):
            with tf.variable_scope('stage_image-feature'):
                output = conv2d_sn(_image, self.nc_f, 3, 3, 1, 1, 'conv', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                output = normalization(output, self.norm, training = training)
                output = activation(output, self.act)

            _curr_dim = output.get_shape().as_list()[1]
            _curr_ch = self.nc_f
            for k in range(0, self.n_shared):
                with tf.variable_scope('stage_conv_shared-{}-{}'.format(_curr_dim, _curr_dim // 2)):
                    with tf.variable_scope('conv_0'):
                        output = conv2d_sn(output, _curr_ch * 2, 3, 3, 2, 2, 'conv', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                        output = normalization(output, self.norm, training = training)
                        output = activation(output, self.act)

                    _curr_dim = _curr_dim // 2
                    _curr_ch = _curr_ch * 2
            

            output_head_list = []
            for k in range(0, self.n_head):
                output_head = output
                with tf.variable_scope('head_{}'.format(k)):
                    _head_dim = _curr_dim
                    _head_ch = _curr_ch
                    while(_head_dim > self.nd_l):
                        with tf.variable_scope('stage_conv_{}-{}'.format(_head_dim, _head_dim // 2)):
                            with tf.variable_scope('conv_0'):
                                output_head = conv2d_sn(output_head, _head_ch * 2, 3, 3, 2, 2, 'conv', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                                output_head = normalization(output_head, self.norm, training = training)
                                output_head = activation(output_head, self.act)                            

                            _head_dim = _head_dim // 2
                            _head_ch = _head_ch * 2

                    with tf.variable_scope('linear'):
                        output_head = tf.layers.flatten(output_head)
                        output_head = linear_sn(output_head, 1, name='linear', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))

                    output_head_list.append(output_head)    #N*1

            output_all = tf.concat(output_head_list, axis = -1)    #N*n_head
            #gather the correct branch for each view
            _batch_size = output_all.get_shape().as_list()[0]
            _out_flatten = tf.reshape(output_all, [_batch_size * self.n_head, 1])       #(N*n_head, 1)
            _delta = tf.range(0, limit=_batch_size * self.n_head, delta = self.n_head, dtype = tf.int64)    #N
            _indice_1d = _view + _delta
            output_final = tf.gather(_out_flatten, _indice_1d)

        self.inited = True
        return output_final

class image_classifier:
    def __init__(self, id = 0, **kwargs):
        self.d_id = id
        self.inited = False

        cfg = kwargs['cfg_args']

        self.nc_f = cfg.nc_f
        self.nd_l = cfg.nd_l
        self.norm = cfg.norm
        self.act = cfg.act
        self.n_head = cfg.n_head

    def forward(self, _image, training = True):
        with tf.variable_scope('classifier_{}'.format(self.d_id), reuse = self.inited):
            with tf.variable_scope('stage_image-feature'):
                output = conv2d_sn(_image, self.nc_f, 3, 3, 1, 1, 'conv', use_sn = False)
                output = normalization(output, self.norm, training = training)
                output = activation(output, self.act)

            _curr_dim = output.get_shape().as_list()[1]
            _curr_ch = self.nc_f

            while(_curr_dim > self.nd_l):
                with tf.variable_scope('stage_conv_{}-{}'.format(_curr_dim, _curr_dim // 2)):
                    output = conv2d_sn(output, _curr_ch * 2, 3, 3, 2, 2, 'conv', use_sn = False)
                    output = normalization(output, self.norm, training = training)
                    output = activation(output, self.act)

                    _curr_dim = _curr_dim // 2
                    _curr_ch = _curr_ch * 2

            with tf.variable_scope('linear'):
                output = tf.layers.flatten(output)
                output = linear_sn(output, self.n_head, name='linear', use_sn = False)          

        self.inited = True
        return output

class voxel_discriminator:
    def __init__(self, id = 0, **kwargs): #nc_f = 32, nd_l = 4, n_shared = 3, n_head = 8, sn = 1, norm = 'None', act = 'relu':
        self.d_id = id
        self.inited = False

        cfg = kwargs['cfg_args']

        self.nc_f = cfg.nc_f
        self.nd_l = cfg.nd_l
        self.sn = cfg.sn
        self.norm = cfg.norm
        self.act = cfg.act

    def forward(self, _voxel, training = True):
        with tf.variable_scope('dis_{}'.format(self.d_id), reuse = self.inited):
            with tf.variable_scope('stage_voxel-feature'):
                output = conv3d_sn(_voxel, self.nc_f, 3, 3, 1, 1, 'conv', use_sn = self.sn, initializer = tf.initializers.truncated_normal(stddev=0.02))
                output = normalization(output, self.norm, training = training)
                output = activation(output, self.act)

            _curr_dim = output.get_shape().as_list()[1]
            _curr_ch = self.nc_f

            while(_curr_dim > self.nd_l):
                with tf.variable_scope('stage_conv_{}-{}'.format(_curr_dim, _curr_dim // 2)):
                    output = conv3d_sn(output, _curr_ch * 2, 3, 3, 2, 2, 'conv', use_sn = self.sn)
                    output = normalization(output, self.norm, training = training)
                    output = activation(output, self.act)

                    _curr_dim = _curr_dim // 2
                    _curr_ch = _curr_ch * 2

            with tf.variable_scope('linear'):
                output = tf.layers.flatten(output)
                output = linear_sn(output, self.n_head, name='linear', use_sn = self.sn)     

        self.inited = True
        return output