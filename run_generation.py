import tensorflow as tf
import numpy as np

from framework.arg_parser.parser_base import parser
from framework.cfg_parser.cfg_mp_gan import MPGANRunnerConfigurator

from framework.graph_runner import mp_gan as _runner_gan
from framework.utils.io import make_dir, write_vox

import os

#adding parameters
parser.add_argument('--model_ckpt', help='model checkpoints', default = './ckpt', type = str)
parser.add_argument('--out_folder', help='root dir for output', default = './results', type = str)
parser.add_argument('--category', help='category to generate (chair_32_syn | bird_64_real)', type = str)
parser.add_argument('--number_to_gen', help='number of total iterations', default = 100, type = int)

parser.add_argument('--gpu_id', help='gpu list', default = '0', type = str)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = 'PCI_BUS_ID'

    #parse arguments
    args_dict = vars(parser.parse_args())
    config_file = './experiment_setup/' + args_dict['category'] + '/gan.yaml'

    print('Initialzing...')
    make_dir(args_dict['out_folder'])

    print('Parsing YAML...')
    #parse yaml for GAN
    gpu_list = list(map(int, args_dict['gpu_id'].split(',')))
    cfg_instance_gan = MPGANRunnerConfigurator()
    cfg_instance_gan.load_from_yaml(config_file, shared_scope = 'shared', additional_shared_dict = {'gpu_list': gpu_list})
    cfg_instance_gan.output_dir = args_dict['out_folder']
    cfg_instance_gan.log_dir = args_dict['out_folder']

    runner_gan = getattr(_runner_gan, cfg_instance_gan.type)()
    runner_gan.load_cfg(cfg_instance_gan)
    #build graph
    runner_gan.build_graph(is_train = False)

    runner_gan.init_session()
    runner_gan.load_previous_model(model = args_dict['model_ckpt'])

    print('Running Generation...')
    voxel_z, voxel_array = runner_gan.run_generation_voxel(model_loaded = True, n_gen = args_dict['number_to_gen'], return_z = True)
    for i in range(0, args_dict['number_to_gen']):
        write_vox(args_dict['out_folder'] + r'/{:03d}.vox'.format(i), voxel_array[i,...,0])   

    runner_gan.close_session()

