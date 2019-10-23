import tensorflow as tf
import os, sys
import cv2
import numpy as np

from framework.arg_parser.parser_base import parser
from framework.cfg_parser.cfg_mp_gan import MPGANRunnerConfigurator
from framework.cfg_parser.cfg_mp_classifier import MPClassifierRunnerConfigurator

from framework.graph_runner import mp_gan as _runner_gan
from framework.graph_runner import mp_classifier as _runner_classifier
from framework.utils.io import make_dir, write_vox

from cluster import run_cluster

#adding parameters
parser.add_argument('--output_dir', help='root dir for output', default = 'results', type = str)
parser.add_argument('--data_dir', help='data dir for output', default = 'data', type = str)
parser.add_argument('--n_iter', help='number of total iterations', default = 5, type = int)
parser.add_argument('--n_cluster', help='number of cluster', default = 8, type = int)

parser.add_argument('--gpu_id', help='gpu list', default = '0', type = str)

parser.add_argument('--cfg_file_gan', help='yaml for gan', default = '', type = str)
parser.add_argument('--cfg_file_classifier', help='yaml for classifier', default = '', type = str)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tf_record(writer, _mask, _view_label, _cluster_label):
    _mask[_mask > 0.5] = 1
    _mask[_mask <= 0.5] = 0
    _, raw_byte = cv2.imencode(".png", _mask * 255)
    raw_byte = raw_byte.tostring()

    d_feature = {
        'raw_byte': _bytes_feature(raw_byte),
        'view_label': _int64_feature(_view_label),
        'cluster_label': _int64_feature(_cluster_label)
    }  

    features = tf.train.Features(feature = d_feature)
    example = tf.train.Example(features = features)
    serialized = example.SerializeToString()
    writer.write(serialized)    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = 'PCI_BUS_ID'

    #parse arguments
    args_dict = vars(parser.parse_args())
    cfg_file_gan = args_dict['cfg_file_gan']
    cfg_file_classifier = args_dict['cfg_file_classifier']

    print('Initialzing...')
    make_dir(args_dict['output_dir'])

    print('Parsing YAML...')
    #parse yaml for GAN
    #overrde gpu setting
    gpu_list = list(map(int, args_dict['gpu_id'].split(',')))
    cfg_instance_gan = MPGANRunnerConfigurator()
    cfg_instance_gan.load_from_yaml(cfg_file_gan, shared_scope = 'shared', additional_shared_dict = {'gpu_list': gpu_list})

    cfg_instance_gan.auto_restart = args_dict['auto_restart']
    cfg_instance_gan.max_num_checkpoint = args_dict['max_num_checkpoint']
    cfg_instance_gan.debug_tag = args_dict['debug_tag']
    cfg_instance_gan.discriminator.network.n_head = args_dict['n_cluster']

    #parse yaml for classifier
    cfg_instance_classifier = MPClassifierRunnerConfigurator()
    cfg_instance_classifier.load_from_yaml(cfg_file_classifier, shared_scope = 'shared', additional_shared_dict = {'gpu_list': gpu_list})
    cfg_instance_classifier.auto_restart = args_dict['auto_restart']
    cfg_instance_classifier.max_num_checkpoint = args_dict['max_num_checkpoint']
    cfg_instance_classifier.debug_tag = args_dict['debug_tag']
    #override gpu setting
    cfg_instance_classifier.gpu_list = list(map(int, args_dict['gpu_id'].split(',')))


    runner_gan = getattr(_runner_gan, cfg_instance_gan.type)()
    runner_classifier = getattr(_runner_classifier, cfg_instance_classifier.type)()

    for i in range(args_dict['n_iter']):
        print('GAN ITERATION {} / {}'.format(i, args_dict['n_iter']))
        #making dirs
        make_dir(args_dict['output_dir'] + r'/gan_iter_{}'.format(i))
        make_dir(args_dict['output_dir'] + r'/view_iter_{}'.format(i))
        make_dir(args_dict['output_dir'] + r'/intermediate_data/gen_{}'.format(i))
        make_dir(args_dict['output_dir'] + r'/intermediate_data/pred_{}'.format(i))

        #STEP 1: Training MP-GAN
        #manually set up input and output dirs
        cfg_instance_gan.output_dir = args_dict['output_dir'] + r'/gan_iter_{}'.format(i)
        cfg_instance_gan.log_dir = cfg_instance_gan.output_dir
        if(i == 0):
            #in beginning: (1) n_dis = 1; (2) data input as origin data input
            cfg_instance_gan.discriminator.network.n_head = 1
            cfg_instance_gan.data_loader_train.data_folder = args_dict['data_dir']
        else:
            #(2) set n_cluster; (2) data input as latest (with predicted view labels)
            cfg_instance_gan.discriminator.network.n_head = args_dict['n_cluster']
            cfg_instance_gan.data_loader_train.data_folder = args_dict['output_dir'] + r'/intermediate_data/pred_{}'.format(i-1)
        
        runner_gan.load_cfg(cfg_instance_gan)
        #build graph
        runner_gan.build_graph(add_inference_part = True)
        #init tf session
        runner_gan.init_session()    
        #after 2nd mp gan training, fintune from previous model
        if(i > 1):
            runner_gan.load_previous_model(model = args_dict['output_dir'] + r'/gan_iter_{}/model-{}'.format(i-1, cfg_instance_gan.max_iter - 1))
        else:
            #still accepts auto-restarting
            runner_gan.load_previous_model()
        #run training
        runner_gan.run_training(model_loaded = True)
        
        #output intermediate data name
        batch_image, batch_label = runner_gan.run_generation_image(model_loaded = True, n_gen = 64000)

        out_filename = args_dict['output_dir'] + r'/intermediate_data/gen_{}/{}'.format(i, cfg_instance_classifier.data_loader_train.data_name)
        writer = tf.python_io.TFRecordWriter(out_filename)
        for _k in range(0, 60000):
            write_tf_record(writer, batch_image[_k], batch_label[_k], 0)
        writer.close()

        out_filename = args_dict['output_dir'] + r'/intermediate_data/gen_{}/{}'.format(i, cfg_instance_classifier.data_loader_val.data_name)
        writer = tf.python_io.TFRecordWriter(out_filename)
        for _k in range(60000, batch_image.shape[0]):
            write_tf_record(writer, batch_image[_k], batch_label[_k], 0)
        writer.close()

        runner_gan.close_session()

        n_view_bin = runner_gan.global_data_dict['n_view_bin']

        #STEP 2: Training Classifier
        #training classifier
        #manually set up input and output dirs
        print('CLASSIFICATION ITERATION {} / {}'.format(i, args_dict['n_iter']))
        cfg_instance_classifier.output_dir = args_dict['output_dir'] + r'/view_iter_{}'.format(i)
        cfg_instance_classifier.log_dir = cfg_instance_classifier.output_dir
        cfg_instance_classifier.data_loader_train.data_folder = args_dict['output_dir'] + r'/intermediate_data/gen_{}'.format(i)
        cfg_instance_classifier.data_loader_val.data_folder = args_dict['output_dir'] + r'/intermediate_data/gen_{}'.format(i)
        cfg_instance_classifier.data_loader_test.data_folder = args_dict['data_dir']  
        cfg_instance_classifier.classifier.network.n_head = n_view_bin

        runner_classifier.load_cfg(cfg_instance_classifier)
        #build graph
        runner_classifier.build_graph()
        #init tf session
        runner_classifier.init_session()
        #run training   
        runner_classifier.run_training()
        #run inference, cluster, write results        
        _image_array, _prob_array, _view_label_array = runner_classifier.run_inference(model_loaded = True, return_whole_dataset = True)

        runner_classifier.close_session()

        #cluster (optional)
        _cluster_label_array = run_cluster(_prob_array, args_dict['n_cluster'])       
        
        #write results
        out_filename = args_dict['output_dir'] + r'/intermediate_data/pred_{}/{}'.format(i, cfg_instance_gan.data_loader_train.data_name)
        writer = tf.python_io.TFRecordWriter(out_filename)
        for i in range(_view_label_array.shape[0]):
            write_tf_record(writer, _image_array[i], _view_label_array[i], _cluster_label_array[i])
        writer.close()