import tensorflow as tf
import numpy as np
import os

from ...utils.log import log_message

from ..module_base import module_base
from ..codebase.io import parse_image_view_cluster_label_from_tfrecord
from ..codebase.visualization import tileImage

class data_loader_mask_label(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']

        log_message('data_loader_mask_label', '---Building subgraph...---')

        if(phase == 'train'):
            gpu_list = cfg_args.gpu_list
            batchsize_gpu = cfg_args.batch_size // len(gpu_list)
        else:
            gpu_list = [0]
            batchsize_gpu = 1

        data_name = os.path.join(cfg_args.data_folder, cfg_args.data_name)

        with tf.name_scope('dataset_{}'.format(phase)):
            dataset = tf.data.TFRecordDataset(data_name)
            if(cfg_args.random_shuffle == 1 and phase == 'train'):
                dataset = dataset.shuffle(8192)
            
            if(phase == 'train'):
                dataset = dataset.repeat()
            dataset = dataset.map(parse_image_view_cluster_label_from_tfrecord, num_parallel_calls = 16)

            if(phase == 'train'):
                dataset = dataset.batch(cfg_args.batch_size)
            else:
                dataset = dataset.batch(1)
                
            dataset = dataset.prefetch(4)

            if(phase == 'train'):
                batch_mask, batch_view_label, batch_cluster_label = dataset.make_one_shot_iterator().get_next()
            else:
                _iterator = dataset.make_initializable_iterator()
                batch_mask, batch_view_label, batch_cluster_label = _iterator.get_next()

            #Need to manually set the shape.
            #I hate tensorflow...
            batch_mask.set_shape([cfg_args.batch_size, cfg_args.height, cfg_args.width, cfg_args.channel])
            batch_view_label.set_shape([cfg_args.batch_size])
            batch_cluster_label.set_shape([cfg_args.batch_size])
            
            if(cfg_args.noise_to_mask > 0 and phase == 'train'):
                _random_noise = tf.random_uniform(shape = batch_mask.get_shape().as_list(), minval = 0.0, maxval = cfg_args.noise_to_mask)
                batch_soft_mask = tf.where(batch_mask > 0.5, batch_mask - _random_noise, batch_mask + _random_noise)
            else:
                batch_soft_mask = batch_mask

            tf_data_all_gpu = []
            tf_view_label_all_gpu = []
            tf_cluster_label_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    tf_data = batch_soft_mask[g_id*batchsize_gpu:(g_id+1)*batchsize_gpu]
                    tf_view_label = batch_view_label[g_id*batchsize_gpu:(g_id+1)*batchsize_gpu]
                    tf_cluster_label = batch_cluster_label[g_id*batchsize_gpu:(g_id+1)*batchsize_gpu]
                    tf_data_all_gpu.append(tf_data)
                    tf_view_label_all_gpu.append(tf_view_label)
                    tf_cluster_label_all_gpu.append(tf_cluster_label)

            self.public_ops['input_mask'] = tf_data_all_gpu
            self.public_ops['input_view_label'] = tf_view_label_all_gpu
            self.public_ops['input_cluster_label'] = tf_cluster_label_all_gpu

            if(phase != 'train'):
                self.public_ops['reset_op'] = _iterator.initializer

        if(phase == 'train'):
            vis_input_data = tileImage(batch_mask, nCol=8)
            self.add_summary(vis_input_data, 'train', 'image', 'input_mask')

class data_loader_voxel(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']

        log_message('data_loader_voxel', '---Building subgraph...---')

        if(phase == 'train'):
            gpu_list = cfg_args.gpu_list
            batchsize_gpu = cfg_args.batch_size // len(gpu_list)
        else:
            gpu_list = [0]
            batchsize_gpu = 1

        data_name = os.path.join(cfg_args.data_folder, cfg_args.data_name)

        with tf.name_scope('dataset_{}'.format(phase)):
            dataset = tf.data.TFRecordDataset(data_name)
            if(cfg_args.random_shuffle == 1 and phase == 'train'):
                dataset = dataset.shuffle(8192)
            
            if(phase == 'train'):
                dataset = dataset.repeat()
            dataset = dataset.map(parse_voxel_from_tfrecord, num_parallel_calls = 16)

            if(phase == 'train'):
                dataset = dataset.batch(cfg_args.batch_size)
            else:
                dataset = dataset.batch(1)
                
            dataset = dataset.prefetch(4)

            if(phase == 'train'):
                batch_voxel = dataset.make_one_shot_iterator().get_next()
            else:
                _iterator = dataset.make_initializable_iterator()
                batch_voxel = _iterator.get_next()

            #Need to manually set the shape.
            #I hate tensorflow...
            batch_voxel = tf.reshape(batch_voxel, [cfg_args.batch_size, cfg_args.height, cfg_args.width, cfg_args.depth, cfg_args.channel])

            if(cfg_args.noise_to_mask > 0 and phase == 'train'):
                _random_noise = tf.random_uniform(shape = batch_voxel.get_shape().as_list(), minval = 0.0, maxval = cfg_args.noise_to_mask)
                batch_soft_voxel = tf.where(batch_voxel > 0.5, batch_voxel - _random_noise, batch_voxel + _random_noise)
            else:
                batch_soft_voxel = batch_voxel

            tf_data_all_gpu = []
            tf_view_label_all_gpu = []
            tf_cluster_label_all_gpu = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    tf_data = batch_soft_voxel[g_id*batchsize_gpu:(g_id+1)*batchsize_gpu]
                    tf_data_all_gpu.append(tf_data)

            self.public_ops['input_voxel'] = tf_data_all_gpu

            if(phase != 'train'):
                self.public_ops['reset_op'] = _iterator.initializer