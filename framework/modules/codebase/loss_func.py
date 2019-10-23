import tensorflow as tf
import math
import numpy as np

from .layers import tensor_norm


def loss_kl(mu, log_square_sd):
    _kl = 0.5 * tf.reduce_mean(tf.exp(log_square_sd) + tf.square(mu) - log_square_sd - 1)
    _kl_loss = _kl#tf.reduce_mean(_kl)
    return _kl_loss   

def loss_wgan_g(fake_d):
    return -tf.reduce_mean(fake_d)

def loss_wgan_d(real_d, fake_d):
    return tf.reduce_mean(fake_d - real_d)

def loss_hinge_d(real_d, fake_d):
    return tf.reduce_mean(tf.nn.relu(1.0 - real_d)) + tf.reduce_mean(tf.nn.relu(1.0 + fake_d))

def loss_gp(interpolate_data, interpolate_d):
    gradients = tf.gradients(interpolate_d, interpolate_data)[0]
    gradient_squares = tf.reduce_sum(tf.square(gradients), reduction_indices = list(range(1, gradients.shape.ndims)))
    slope = tf.sqrt(gradient_squares + 1e-10)
    penalties = slope - 1.0
    return tf.reduce_mean(tf.square(penalties))

def loss_l1(data_1, data_2):
    return tf.reduce_mean(tf.abs(data_1 - data_2))

def loss_l2(data_1, data_2):
    return tf.reduce_mean((data_1 - data_2) ** 2)

def loss_tv_voxel(data, log_transform = False, method = 'mean'):
    if(log_transform):
        data = tf.clip_by_value(data, 1e-6, 1.0)
        data = tf.math.log(data)

    ndims = data.get_shape().ndims
    if(ndims == 4):
        data = data[tf.newaxis,...]
        
    pixel_dif1 = data[:, 1:, :, :, :] - data[:, :-1, :, :, :]
    pixel_dif2 = data[:, :, 1:, :] - data[:, :, :-1, :, :]
    pixel_dif3 = data[:, :, :, 1:, :] - data[:, :, :, :-1, :]

    if(method == 'mean'):
        tv = tf.reduce_mean(tf.math.abs(pixel_dif1)) + tf.reduce_mean(tf.math.abs(pixel_dif2)) + tf.reduce_mean(tf.math.abs(pixel_dif3))
    elif(method == 'sum'):
        tv = tf.reduce_sum(tf.math.abs(pixel_dif1)) + tf.reduce_sum(tf.math.abs(pixel_dif2)) + tf.reduce_sum(tf.math.abs(pixel_dif3))
    
    return tv

def loss_beta_prior(data):
    data = tf.clip_by_value(data, 1e-6, 1.0 - 1e-6)
    _prior = tf.reduce_mean(tf.math.log(data) + tf.math.log(1.0 - data))
    return _prior


def loss_tv(data, log_transform = False, method = 'mean'):
    if(log_transform):
        data = tf.clip_by_value(data, 1e-6, 1.0)
        data = tf.math.log(data)

    data_shape = data.get_shape().as_list()
    if(method == 'mean'):
        _n = data_shape[0] * data_shape[1] * data_shape[2] * data_shape[3]
    elif(method == 'sum'):
        _n = data_shape[0]

    return tf.reduce_sum(tf.image.total_variation(data)) / _n