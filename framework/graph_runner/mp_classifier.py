import tensorflow as tf
import numpy as np
import os, sys
from tqdm import tqdm

from ..utils.log import log_message
from .graph_runner_base import graph_runner

from ..modules.data_loader import mp_gan as data_loader
from ..modules.network import mp_gan as network
from ..modules.loss import mp_gan as loss
from ..modules.solver import solver_base as solver

class mp_classifier(graph_runner):
    def __init__(self, cfg_args = None):
        super().__init__(cfg_args)

    def build_graph(self):
        super().build_graph()
        log_message('mp_classifier', '-----Building Graph...-----')
        #shared class or ops
        global_data_dict = {}

        with self.graph.as_default():
            #do we have training task?            
            _training_tag = False if self.cfg_args.data_loader_train.type == 'default' else True
            _inference_tag = False if self.cfg_args.data_loader_test.type == 'default' else True

            if(_training_tag):
                #data loader submodule
                log_message('mp_classifier', '-----Building data loader...-----')
                data_module_train = getattr(data_loader, self.cfg_args.data_loader_train.type)()
                data_module_train.build_graph(
                    cfg_args = self.cfg_args.data_loader_train, 
                    phase = 'train',
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['data_loader_train'] = data_module_train 

                data_module_val = getattr(data_loader, self.cfg_args.data_loader_val.type)()
                data_module_val.build_graph(
                    cfg_args = self.cfg_args.data_loader_val, 
                    phase = 'val',
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['data_loader_val'] = data_module_val

                log_message('mp_classifier', '-----Building classifier...-----')
                #classifier submodule
                classifier_module = getattr(network, self.cfg_args.classifier.type)()
                classifier_module.build_graph(
                    cfg_args = self.cfg_args.classifier, 
                    phase = 'train',
                    data_loader = data_module_train,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['classifier'] = classifier_module
                
                classifier_module_val = getattr(network, self.cfg_args.classifier.type)()
                classifier_module_val.build_graph(
                    cfg_args = self.cfg_args.classifier, 
                    phase = 'val',
                    data_loader = data_module_val,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['classifier_val'] = classifier_module_val

                log_message('mp_classifier', '-----Building loss...-----')
                #loss submodule
                loss_module = getattr(loss, self.cfg_args.loss.type)()
                loss_module.build_graph(
                    cfg_args = self.cfg_args.loss,
                    phase = 'train',
                    data_loader = data_module_train,
                    classifier = classifier_module,
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['loss'] = loss_module

                loss_module_val = getattr(loss, self.cfg_args.loss.type)()
                loss_module_val.build_graph(
                    cfg_args = self.cfg_args.loss,
                    phase = 'val',
                    data_loader = data_module_val,
                    classifier = classifier_module_val,
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['loss_val'] = loss_module_val
            
                log_message('mp_classifier', '-----Building solver...-----')
                #solver submodule
                solver_module = getattr(solver, self.cfg_args.solver.type)()
                loss_list = [self.submodule_dict['loss'].public_ops['loss_full']]
                solver_module.build_graph(
                    cfg_args = self.cfg_args.solver,
                    loss_list = loss_list,
                    var_prefix = 'classifier_'
                )
                self.submodule_dict['solver'] = solver_module

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

            if(_inference_tag):
                log_message('mp_classifier', '-----Building test data loader...-----')
                data_module_inference = getattr(data_loader, self.cfg_args.data_loader_test.type)()
                data_module_inference.build_graph(
                    cfg_args = self.cfg_args.data_loader_test, 
                    phase = 'test',
                    global_data_dict = global_data_dict
                )
                self.submodule_dict['data_loader_inference'] = data_module_inference   

                log_message('mp_classifier', '-----Building test classifier...-----')
                classifier_module_inference = getattr(network, self.cfg_args.classifier.type)()
                classifier_module_inference.build_graph(
                    cfg_args = self.cfg_args.classifier, 
                    phase = 'test',
                    data_loader = data_module_inference,
                    global_data_dict = global_data_dict            
                )
                self.submodule_dict['classifier_inference'] = classifier_module_inference                

    def run_training(self, model_loaded = False):
        #load previous model if have
        if(model_loaded == False):
            _loaded = self.load_previous_model()        

        log_message('mp_classifier', '-----Begin Training...-----')
        #run training
        for it in tqdm(range(self.cfg_args.max_iter), file=sys.stdout):
            self.sess.run(self.submodule_dict['solver'].public_ops['train_op'])

            if(it % self.cfg_args.display_step == 0):
                log_message('mp_classifier', '-----Iter {}/{}-----'.format(it, self.cfg_args.max_iter))
                loss_full = self.sess.run(self.submodule_dict['loss'].public_ops['loss_full'])
                #train_acc = self.sess.run(self.submodule_dict['loss'].public_ops['train_acc'])
                log_message('mp_classifier', 'loss = {:05f}'.format(np.mean(loss_full)))     

                summary_train = self.sess.run(self.summary_train_op)
                self.tensorboard_writer.add_summary(summary_train, it)
                            
            if(it % self.cfg_args.checkpoint_step == 0):
                #do validation
                self.sess.run(self.submodule_dict['data_loader_val'].public_ops['reset_op'])
                self.sess.run(self.submodule_dict['loss_val'].public_ops['reset_acc'])
                try:
                    while True:
                        self.sess.run(self.submodule_dict['loss_val'].public_ops['valid_acc_op'])
                except tf.errors.OutOfRangeError:
                    valid_acc = self.sess.run(self.submodule_dict['loss_val'].public_ops['valid_acc'])
                    log_message('mp_classifier', 'val acc = {:03f}'.format(valid_acc))

                if(self.summary_val_op != None):
                    summary_val = self.sess.run(self.summary_val_op)
                    self.tensorboard_writer.add_summary(summary_val, it)
                self.saver.save(self.sess, self.cfg_args.output_dir + r'/model', global_step = it)

        log_message('mp_classifier', '-----Finished Training...-----')
        self.saver.save(self.sess, self.cfg_args.output_dir + r'/model', global_step = self.cfg_args.max_iter - 1)

    def run_inference(self, model_loaded = False, return_whole_dataset = True, override_input = None):
        if(model_loaded == False):
            _loaded = self.load_previous_model()

        log_message('mp_classifier', '-----Begin Inference...-----')
        if(override_input != None):
            _raw_data, _prob = self.sess.run([
                    self.submodule_dict['data_loader_inference'].public_ops['input_mask'][0],
                    self.submodule_dict['classifier_inference'].public_ops['predict_prob'][0]
                ],
                feed_dict = {
                    self.submodule_dict['data_loader_inference'].public_ops['input_mask'][0]: override_input
                }
            )
            _raw_data = _raw_data[0]
            _prob = _prob.flatten()
            _label = np.argmax(_prob)     

            return _raw_data, _prob, _label
               
        #run inference
        self.sess.run(self.submodule_dict['data_loader_inference'].public_ops['reset_op'])
        _data_list = []
        _prob_list = []
        _label_list = []
        try:
            while True:
                _raw_data, _prob = self.sess.run([
                        self.submodule_dict['data_loader_inference'].public_ops['input_mask'][0],
                        self.submodule_dict['classifier_inference'].public_ops['predict_prob'][0]
                    ])
                
                _raw_data = _raw_data[0]
                _prob = _prob.flatten()
                _label = np.argmax(_prob)

                if(return_whole_dataset):
                    _data_list.append(_raw_data)
                    _prob_list.append(_prob)
                    _label_list.append(_label)
                else:
                    return _raw_data, _prob, _label
        except tf.errors.OutOfRangeError:
            log_message('mp_classifier', '-----Finished Inference...-----')
            if(return_whole_dataset):
                return np.stack(_data_list, axis = 0), np.stack(_prob_list, axis = 0), np.stack(_label_list, axis = 0)
            else:
                return None, None, None


    