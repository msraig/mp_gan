import tensorflow as tf
import numpy as np
import os, sys
from tqdm import tqdm

from ..utils.log import log_message
from ..utils.io import make_dir, write_vox
from .graph_runner_base import graph_runner

from ..modules.data_loader import mp_gan as data_loader
from ..modules.network import mp_gan as network
from ..modules.projection import mp_gan as projection
from ..modules.loss import mp_gan as loss
from ..modules.solver import solver_base as solver

class mp_gan(graph_runner):
    def __init__(self, cfg_args = None):
        super().__init__(cfg_args)

    def build_graph(self, is_train = True, add_inference_part = False):
        super().build_graph()
        log_message('mp_gan', '-----Building Graph...-----')
        #shared class or ops
        global_data_dict = {}

        if(is_train == False):
            add_inference_part = True

        with self.graph.as_default():
            #data loader submodule
            if(is_train):
                log_message('mp_gan', '-----Building data loader...-----')
                data_module_train = getattr(data_loader, self.cfg_args.data_loader_train.type)()
                data_module_train.build_graph(
                    cfg_args = self.cfg_args.data_loader_train, 
                    phase = 'train',
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['data_loader_train'] = data_module_train
            
                log_message('mp_gan', '-----Building generator...-----')
                #generator submodule
                generator_module = getattr(network, self.cfg_args.generator.type)()
                generator_module.build_graph(
                    cfg_args = self.cfg_args.generator, 
                    phase = 'train',
                    data_loader = data_module_train,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['generator'] = generator_module
            
                log_message('mp_gan', '-----Building projection...-----')
                #projection submodule
                projection_module = getattr(projection, self.cfg_args.projection.type)()
                projection_module.build_graph(
                    cfg_args = self.cfg_args.projection, 
                    phase = 'train',
                    generator = generator_module,
                    data_loader = data_module_train,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['projection'] = projection_module

                log_message('mp_gan', '-----Building discriminator...-----')
                #discriminator submodule
                discriminator_module = getattr(network, self.cfg_args.discriminator.type)()
                discriminator_module.build_graph(
                    cfg_args = self.cfg_args.discriminator, 
                    phase = 'train',
                    data_loader = data_module_train,
                    projection = projection_module,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['discriminator'] = discriminator_module

                log_message('mp_gan', '-----Building loss...-----')
                #loss submodule
                loss_module = getattr(loss, self.cfg_args.loss.type)()
                loss_module.build_graph(
                    cfg_args = self.cfg_args.loss,
                    phase = 'train',
                    data_loader = data_module_train,
                    generator = generator_module,
                    projection = projection_module,
                    discriminator = discriminator_module,
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['loss'] = loss_module
                
                log_message('mp_gan', '-----Building solver G...-----')
                #G solver submodule
                loss_list_g = [self.submodule_dict['loss'].public_ops['loss_g']]
                if('loss_tv_voxel' in self.submodule_dict['loss'].public_ops):
                    loss_list_g.append(self.submodule_dict['loss'].public_ops['loss_tv_voxel'])
                if('loss_beta_prior' in self.submodule_dict['loss'].public_ops):
                    loss_list_g.append(self.submodule_dict['loss'].public_ops['loss_beta_prior'])

                solver_module_G = getattr(solver, self.cfg_args.solver_G.type)()
                solver_module_G.build_graph(
                    cfg_args = self.cfg_args.solver_G,
                    loss_list = loss_list_g,
                    var_prefix = 'gen_'
                )
                self.submodule_dict['solver_g'] = solver_module_G

                log_message('mp_gan', '-----Building solver D...-----')
                #D solver submodule
                loss_list_d = [self.submodule_dict['loss'].public_ops['loss_d']]
                if('loss_gp' in self.submodule_dict['loss'].public_ops):
                    loss_list_d.append(self.submodule_dict['loss'].public_ops['loss_gp'])

                solver_module_D = getattr(solver, self.cfg_args.solver_D.type)()
                solver_module_D.build_graph(
                    cfg_args = self.cfg_args.solver_D,
                    loss_list = loss_list_d,
                    var_prefix = 'dis_'
                )
                self.submodule_dict['solver_d'] = solver_module_D

                #collect all tensoboard ops
                tensorboard_train = []
                tensorboard_val = []
                for _name in self.submodule_dict:
                    tensorboard_train += self.submodule_dict[_name].tensorboard_ops['train']
                    tensorboard_val += self.submodule_dict[_name].tensorboard_ops['val']

                self.summary_train_op = tf.summary.merge(tensorboard_train)
                if(tensorboard_val != []):
                    self.summary_val_op = tf.summary.merge(tensorboard_val)
                else:
                    self.summary_val_op = None

            if(add_inference_part):
                log_message('mp_gan', '-----Building inference generator...-----')
                #generator submodule
                generator_module_inference = getattr(network, self.cfg_args.generator.type)()
                generator_module_inference.build_graph(
                    cfg_args = self.cfg_args.generator, 
                    phase = 'test',
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['generator_inference'] = generator_module_inference
            
                log_message('mp_gan', '-----Building inference projection...-----')
                #projection submodule
                projection_module_inference = getattr(projection, self.cfg_args.projection.type)()
                projection_module_inference.build_graph(
                    cfg_args = self.cfg_args.projection, 
                    phase = 'test',
                    generator = generator_module_inference,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['projection_inference'] = projection_module_inference

            self.global_data_dict = global_data_dict     

    def run_training(self, model_loaded = False):
        #load previous auto-restart model if have
        if(model_loaded == False):
            _loaded = self.load_previous_model()

        make_dir(self.cfg_args.output_dir + r'/debug_shape')

        log_message('mp_gan', '-----Begin Training...-----')
        #run training
        for it in tqdm(range(self.cfg_args.max_iter), file=sys.stdout):
            for it_d in range(0, self.cfg_args.solver_D.dg_ratio):
                self.sess.run(self.submodule_dict['solver_d'].public_ops['train_op'])
            for it_g in range(0, self.cfg_args.solver_G.dg_ratio):
                self.sess.run(self.submodule_dict['solver_g'].public_ops['train_op'])

            if(it % self.cfg_args.display_step == 0):
                log_message('mp_gan', '-----Iter {}/{}-----'.format(it, self.cfg_args.max_iter))
                loss_g = self.sess.run(self.submodule_dict['loss'].public_ops['loss_g'])
                loss_d = self.sess.run(self.submodule_dict['loss'].public_ops['loss_d'])
                log_message('mp_gan', 'loss_g = {:05f}, loss_d = {:05f}'.format(np.mean(loss_g), np.mean(loss_d)))              

                summary_train = self.sess.run(self.summary_train_op)
                self.tensorboard_writer.add_summary(summary_train, it)

                if('generator_inference' in self.submodule_dict):
                    _gen_voxel = self.run_generation_voxel(model_loaded = True, n_gen = 64)
                    for i in range(0, 64):
                        write_vox(self.cfg_args.output_dir + r'/debug_shape/{:03d}.vox'.format(i), _gen_voxel[i,...,0])
                            
            if(it % self.cfg_args.checkpoint_step == 0):
                self.saver.save(self.sess, self.cfg_args.output_dir + r'/model', global_step = it)

        log_message('mp_gan', '-----Finished Training...-----')
        self.saver.save(self.sess, self.cfg_args.output_dir + r'/model', global_step = self.cfg_args.max_iter - 1)

    def run_generation_voxel(self, model_loaded = False, n_gen = 100, return_z = False):
        #load previous auto-restart model if have
        if(model_loaded == False):
            _loaded = self.load_previous_model()

        log_message('mp_gan', '-----Begin Generate Voxels...-----')
        output_voxel_list = []
        output_z_list = []
        n_batch = n_gen // (16 * len(self.cfg_args.gpu_list)) + 1
        for i in range(n_batch):
            _voxel_z, _voxel_gen = self.sess.run([self.submodule_dict['generator_inference'].public_ops['random_z'], self.submodule_dict['generator_inference'].public_ops['voxel_generated']])
            _voxel_z = np.concatenate(_voxel_z, axis = 0)
            _voxel_gen = np.concatenate(_voxel_gen, axis = 0)

            output_z_list.append(_voxel_z)
            output_voxel_list.append(_voxel_gen)
        
        output_z_all = np.concatenate(output_z_list, axis = 0)
        output_voxel_all = np.concatenate(output_voxel_list, axis = 0)

        output_z_all = output_z_all[0:n_gen]
        output_voxel_all = output_voxel_all[0:n_gen]
        if(return_z):
            return output_z_all, output_voxel_all
        else:
            return output_voxel_all
        
    def run_generation_image(self, model_loaded = False, n_gen = 100):
        #load previous auto-restart model if have
        if(model_loaded == False):
            _loaded = self.load_previous_model()

        log_message('mp_gan', '-----Begin Generate Images...-----')
        output_image_list = []
        output_label_list = []
        n_batch = n_gen // (16 * len(self.cfg_args.gpu_list)) + 1
        for i in range(n_batch):
            _image_gen, _view_label = self.sess.run(
                [
                    self.submodule_dict['projection_inference'].public_ops['projected_mask'],
                    self.submodule_dict['projection_inference'].public_ops['view_label']
                ]
            )
            _image_gen = np.concatenate(_image_gen, axis = 0)
            _view_label = np.concatenate(_view_label, axis = 0)

            output_image_list.append(_image_gen)
            output_label_list.append(_view_label)
        
        output_image_all = np.concatenate(output_image_list, axis = 0)
        output_label_all = np.concatenate(output_label_list, axis = 0)
        return output_image_all[0:n_gen], output_label_all[0:n_gen]



    