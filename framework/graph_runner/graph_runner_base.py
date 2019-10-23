import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from ..utils.io import make_dir
from ..utils.log import init_logger, log_message

class graph_runner:
    def __init__(self, cfg_args = None):
        if(cfg_args != None):
            self.load_cfg(cfg_args)

        self.submodule_dict = {}        
        self.graph = None
        self.sess = None

    def load_cfg(self, cfg_args):
        self.cfg_args = cfg_args
        #log dir
        make_dir(cfg_args.log_dir)
        log_file = os.path.join(cfg_args.log_dir, 'log.txt')
        init_logger(log_file)
        #experiment dir
        make_dir(cfg_args.output_dir)
        #set gpu id
        log_message('graph_runner', '------Using GPU: {}------'.format(cfg_args.gpu_list))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(str, cfg_args.gpu_list)))        

    def build_graph(self):
        if(self.graph != None):
            del self.graph
        self.graph = tf.Graph()
      
    def init_session(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep = self.cfg_args.max_num_checkpoint)
            self.init_op_global = tf.global_variables_initializer()
            self.init_op_local = tf.local_variables_initializer()

        log_message('graph_runner', '-----Initialize Tensorflow...-----')
        config = tf.ConfigProto()
        config.allow_soft_placement=True
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config, graph=self.graph)
        if(self.cfg_args.debug_tag):
            log_message('graph_runner', '-----TF debug mode enabled.-----')
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        self.tensorboard_writer = tf.summary.FileWriter(self.cfg_args.log_dir, self.sess.graph)

        log_message('graph_runner', '-----Initialize global variables...-----')
        self.sess.run(self.init_op_global)
        self.sess.run(self.init_op_local)

    def load_previous_model(self, model = ''):
        _loaded = False
        if(model != ''):
            self.saver.restore(self.sess, model)
            log_message('graph_runner', '---Loaded previous model {}...----'.format(model))
            _loaded = True
        else:
            if(os.path.exists(self.cfg_args.previous_model + '.meta')):
                log_message('graph_runner', '-----Load previous model (if have)...-----')
                self.saver.restore(self.sess, self.cfg_args.previous_model)
                log_message('graph_runner', '-----Restored {}...-----'.format(self.cfg_args.previous_model))
                _loaded = True
            elif(self.cfg_args.auto_restart):
                latest_checkpoint = tf.train.latest_checkpoint(self.cfg_args.output_dir)
                if(latest_checkpoint != None):
                    checkpoint_iter = int(latest_checkpoint.split('-')[-1])
                    self.saver.restore(self.sess, latest_checkpoint)
                    log_message('graph_runner', '-----Auto restored {}...-----'.format(latest_checkpoint))
                    log_message('graph_runner', '-----Training begin from iter {}...-----'.format(checkpoint_iter))
                    self.cfg_args.max_iter -= checkpoint_iter
                    _loaded = True
                else:
                    log_message('graph_runner', '-----No previous model found.-----')
            else:
                log_message('graph_runner', '-----No previous model found.-----')
        
        return _loaded
            
    def close_session(self):
        if(self.sess != None):
            self.sess.close()
        self.sess = None

    

    

    

    