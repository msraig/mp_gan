import tensorflow as tf
from .solver_base import solver_base
from ...utils.log import log_message

class solver_gan(solver_base):
    def __init__(self):
        solver_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        loss_list = kwargs['loss']
        var_prefix = kwargs['var_prefix']

        log_message('solver_gan', '---Graph Solver..---')

        gpu_list = cfg_args.gpu_list

        tf_vars = tf.trainable_variables()
        var_g = [var for var in tf_vars if 'gen_' in var.name]
        var_d = [var for var in tf_vars if 'dis_' in var.name]

        loss_g_full_list = loss.public_ops['loss_g']
        loss_d_full_list = []
        for g_id in range(0, len(gpu_list)):
            with tf.device('/gpu:{}'.format(g_id)):
                _loss_d = loss.public_ops['loss_d'][g_id]
                if('loss_gp' in loss.public_ops):
                    _loss_d += loss.public_ops['loss_gp'][g_id]

                loss_d_full_list.append(_loss_d)

        train_op_g = self.construct_solver(loss_g_full_list, var_g, cfg_args.param)
        train_op_d = self.construct_solver(loss_d_full_list, var_d, cfg_args.param)

        self.public_ops['train_op_g'] = train_op_g
        self.public_ops['train_op_d'] = train_op_d

class solver_classifier(solver_base):
    def __init__(self):
        solver_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        loss = kwargs['loss']
        global_data_dict = kwargs['global_data_dict']

        log_message('solver_classifier', '---Graph Solver..---')

        gpu_list = cfg_args.gpu_list

        tf_vars = tf.trainable_variables()
        var_ = [var for var in tf_vars if 'classifier_' in var.name]

        loss_full_list = loss.public_ops['loss_full']
        train_op = self.construct_solver(loss_full_list, var_, cfg_args.param)
        self.public_ops['loss_full'] = loss_full_list
        self.public_ops['train_op'] = train_op