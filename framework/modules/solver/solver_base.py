import tensorflow as tf
from ..module_base import module_base
from ...utils.log import log_message

class solver_base(module_base):

    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        loss_list = kwargs['loss_list']
        var_prefix = kwargs['var_prefix']

        log_message('solver_gan', '---Graph Solver..---')

        gpu_list = cfg_args.gpu_list

        tf_vars = tf.trainable_variables()
        if(var_prefix != ''):
            var_list = [var for var in tf_vars if var_prefix in var.name]
        else:
            var_list = tf_vars

        if(len(loss_list) == 1):
            loss_full_list = loss_list[0]
        else:
            loss_full_list = []
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    _loss = sum([_term[g_id] for _term in loss_list])
                loss_full_list.append(_loss)     

        train_op = self.construct_solver(loss_full_list, var_list, cfg_args.param)
        self.public_ops['train_op'] = train_op

    def average_gradient(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, var in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def construct_solver(self, _loss_on_each_gpu, _var_list, solver_param):
        log_message('solver_base', '---Construct Solver..---')
        #parse solver param
        solver_param_list = solver_param.split('-')
        solver_type = solver_param_list[0]
        learning_rate = float(solver_param_list[1])

        with tf.name_scope('Solver'):
            with tf.device('/cpu:0'):
                if(solver_type == 'adam'):
                    beta1, beta2 = float(solver_param_list[2]), float(solver_param_list[3])
                    tf_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta1 = beta1,
                                                beta2 = beta2)
                else:
                    raise NotImplementedError()

            tower_grads = []
            for g_id, _loss in enumerate(_loss_on_each_gpu):
                with tf.device('/gpu:{}'.format(g_id)):
                    grad_tower = tf_solver.compute_gradients(_loss, _var_list)
                    tower_grads.append(grad_tower)

            with tf.device('/cpu:0'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grad_avg = self.average_gradient(tower_grads)
                    apply_gradient_op = tf_solver.apply_gradients(grad_avg)

        return apply_gradient_op











