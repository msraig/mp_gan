import tensorflow as tf
from ..utils.log import log_message

#import plugin for display 3D things
from tensorboard.plugins.mesh import summary as mesh_summary

class module_base:
    def __init__(self):
        super().__init__()

        self.public_ops = {}
        self.tensorboard_ops = {
            'train': [],
            'val': [],
            'test': []
        }
        
    def build_graph(self, **kwargs):
        raise NotImplementedError()

    def add_summary(self, data, phase, data_type, name):
        if(data_type == 'scalar'):
            self.tensorboard_ops[phase].append(tf.summary.scalar(name, data))
            log_message('module_base', '---Scalar {} added to tensorboard.---'.format(name))
        elif(data_type == 'image'):
            self.tensorboard_ops[phase].append(tf.summary.image(name, data))
            log_message('module_base', '---Image {} added to tensorboard.---'.format(name))
        elif(data_type == 'voxel'):
            #dereferenced packed data with "**"
            self.tensorboard_ops[phase].append(mesh_summary.op(name, **data))#vertices = data['vertices'], config_dict = data['config_dict']))
            log_message('module_base', '---Voxel {} added to tensorboard.---'.format(name))
        else:
            raise NotImplementedError()

