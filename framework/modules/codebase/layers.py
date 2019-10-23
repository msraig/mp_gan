import tensorflow as tf

def normalizeInput(input_data):
    return input_data * 2.0 - 1.0

def denormalizeInput(input_data):
    return 0.5 * (input_data + 1.0)

def bbox_normalize(input_data, leading_dim = True):
    min_pos = tf.reduce_min(input_data, axis = -2, keepdims = True)
    max_pos = tf.reduce_max(input_data, axis = -2, keepdims = True)

    input_data = (input_data - min_pos) / (max_pos - min_pos + 1e-6)
    input_data = normalizeInput(input_data)
    return input_data

def interpolate_for_gp(real_data, fake_data):
    batchsize = real_data.get_shape().as_list()[0]
    alpha_shape = [batchsize] + [1] * (real_data.shape.ndims - 1)
    _alpha = tf.random_uniform(shape=alpha_shape)

    differences = fake_data - real_data
    interpolates = real_data + (_alpha * differences)

    return interpolates

def tensor_dot(a,b):
    return tf.reduce_sum(a * b, axis = -1) [..., tf.newaxis]

def tensor_norm(tensor):
    t = tf.identity(tensor)
    Length = tf.sqrt(tf.reduce_sum(tf.square(t), axis = -1, keepdims =True))
    return tf.div(t, Length + 1e-8)

#specturm normalization
#https://github.com/google/compare_gan/blob/master/compare_gan/src/gans/ops.py

def spectral_norm(input_):
    """Performs Spectral Normalization on a weight tensor."""
    if len(input_.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input_, (-1, input_.shape[-1]))

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        input_.name.replace(":", "") + "/u_var",
        shape=(w.shape[0], 1),
        dtype=w.dtype,
        initializer=tf.random_normal_initializer(),
        trainable=False)
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(
            tf.matmul(tf.transpose(w), u), axis=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=1e-12)

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input_.shape)
    return w_tensor_normalized

#gan layers
def linear_sn(input_,
           output_size,
           name='linear',
           initializer=tf.contrib.layers.variance_scaling_initializer(),
           use_sn=True,
           use_bias=True):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable(
            "Matrix", [shape[1], output_size],
            tf.float32,
            initializer)
        if use_sn:
            output = tf.matmul(input_, spectral_norm(matrix))
        else:
            output = tf.matmul(input_, matrix)

        if(use_bias):
            bias = tf.get_variable(
                "bias", [output_size], initializer=tf.constant_initializer(0.0))
            output = output + bias
    return output

def conv2d_sn(input_, output_dim, k_h, k_w, d_h, d_w, name="conv2d",
           initializer=tf.contrib.layers.variance_scaling_initializer(), use_sn=False, use_bias = True):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=initializer)
        if use_sn:
            conv = tf.nn.conv2d(
                input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
        if use_bias:
            biases = tf.get_variable(
                "biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)
        else:
            return conv

def conv3d_sn(input_, output_dim, k_d, k_h, k_w, d_d, d_h, d_w, name="conv3d",
           initializer=tf.contrib.layers.variance_scaling_initializer(), use_sn=False, use_bias = True):
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=initializer)
        if use_sn:
            conv = tf.nn.conv3d(
                input_, spectral_norm(w), strides=[1, d_d, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding="SAME")
        if use_bias:
            biases = tf.get_variable(
                "biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)
        else:
            return conv

def deconv2d_sn(input_, output_dim, k_h, k_w, d_h, d_w, name="deconv2d",
           initializer=tf.contrib.layers.variance_scaling_initializer(), use_sn=False, use_bias = True):

    input_shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_h, k_w, output_dim, input_.get_shape()[-1]],
            initializer=initializer)
        if use_sn:
            conv = tf.nn.conv2d_transpose(
                input_, spectral_norm(w), output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, output_dim], strides=[1, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv2d_transpose(input_, w, output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, output_dim], strides=[1, d_h, d_w, 1], padding="SAME")
        if use_bias:
            biases = tf.get_variable(
                "biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        else:
            return conv

def deconv3d_sn(input_, output_dim, k_d, k_h, k_w, d_d, d_h, d_w, name="deconv3d",
           initializer=tf.contrib.layers.variance_scaling_initializer(), use_sn=False, use_bias = True):
    input_shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_d, k_h, k_w, output_dim, input_shape[-1]],
            initializer=initializer)
        if use_sn:
            conv = tf.nn.conv3d_transpose(
                input_, spectral_norm(w), output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3] * 2, output_dim], strides=[1, d_d, d_h, d_w, 1], padding="SAME")
        else:
            conv = tf.nn.conv3d_transpose(input_, w, output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3] * 2, output_dim], strides=[1, d_d, d_h, d_w, 1], padding="SAME")
        if use_bias:
            biases = tf.get_variable(
                "biases", [output_dim], initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)
        else:
            return conv

#Conditional BatchNorm
#cGAN with Projection discrimitor
def one_hot_embedding(_class_one_hot, feature_dim, use_sn = True):
    with tf.variable_scope('one_hot_embedding'):
        output = linear_sn(_class_one_hot, feature_dim, use_bias = False, use_sn = use_sn)
    return output

#ref: https://github.com/AaronLeong/BigGAN-pytorch/blob/master/model_resnet.py
def conditional_batch_norm(_input, _class, training = True, _debug = False):
    with tf.variable_scope('conditional_bn'):
        n_channal = _input.get_shape().as_list()[-1]
        with tf.variable_scope('raw_bn'):
            _raw_bn = tf.layers.batch_normalization(_input, training = True, center = False, scale = False, fused = True)
        with tf.variable_scope('class_embedding'):
            _embedding_gamma_beta = one_hot_embedding(_class, 2 * n_channal, use_sn = True)#gamma_beta_projection(_class, n_channal, use_sn = False)
            _embedding_gamma = tf.expand_dims(tf.expand_dims(_embedding_gamma_beta[...,0:n_channal], 1), 1)
            _embedding_beta = tf.expand_dims(tf.expand_dims(_embedding_gamma_beta[...,n_channal::], 1), 1)
        with tf.variable_scope('output_bn'):
            _out_bn = _raw_bn * _embedding_gamma + _embedding_beta

        if(_debug):
            return _out_bn, [_embedding_gamma, _embedding_beta]
        else:
            return _out_bn

#Conditional Instance Norm
#ref: https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer
def condition_instance_norm(_input, _class):
    with tf.variable_scope('conditional_in'):
        n_channal = _input.get_shape().as_list()[-1]
        with tf.variable_scope('raw_in'):
            _raw_in = tf.contrib.layers.instance_norm(_input, center = False, scale = False)
        with tf.variable_scope('class_embedding'):
            _embedding_gamma_beta = one_hot_embedding(_class, 2 * n_channal, use_sn = False)#gamma_beta_projection(_class, n_channal, use_sn = False)#one_hot_embedding(_class, n_channal, use_sn = True)#
            _embedding_gamma = tf.expand_dims(tf.expand_dims(_embedding_gamma_beta[...,0:n_channal], 1), 1)
            _embedding_beta = tf.expand_dims(tf.expand_dims(_embedding_gamma_beta[...,n_channal::], 1), 1)
        with tf.variable_scope('output_in'):
            _out_in = _raw_in * _embedding_gamma + _embedding_beta
        return _out_in

def normalization(_input, method, training = True, _class = None):
    output = _input
    if(method == 'bn'):
        output = tf.layers.batch_normalization(_input, training = training, fused = True)

    if(method == 'cbn'):
        output = conditional_batch_norm(_input, _class, training = training)

    if(method == 'ln'):
        output = tf.contrib.layers.layer_norm(_input)

    if(method == 'in'):
        output = tf.contrib.layers.instance_norm(_input)

    if(method == 'cin'):
        output = condition_instance_norm(_input, _class)

    return output


def activation(_input, method):
    output = _input
    if(method == 'relu'):
        output = tf.nn.relu(_input)

    if(method == 'lrelu'):
        output = tf.nn.leaky_relu(_input, alpha = 0.1)

    if(method == 'sigmoid'):
        output = tf.sigmoid(_input)

    if(method == 'tanh'):
        output = tf.tanh(_input)

    return output


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

#https://github.com/taki0112/Self-Attention-GAN-Tensorflow
def attention(input):
    input_shape = input.get_shape().as_list()
    ch = input_shape[-1]

    f = tf.layers.conv2d(input,
        filters = ch // 8,
        kernel_size = 1,
        strides = 1,
        padding = 'same',
        use_bias = False)

    g = tf.layers.conv2d(input,
        filters = ch // 8,
        kernel_size = 1,
        strides = 1,
        padding = 'same',
        use_bias = False)

    g_flatten = hw_flatten(g)
    f_flatten = hw_flatten(f)

    s = tf.matmul(g_flatten, f_flatten, transpose_b = True)

    h = tf.layers.conv2d(input,
        filters = ch,
        kernel_size = 1,
        strides = 1,
        padding = 'same',
        use_bias = False)

    beta = tf.nn.softmax(s, axis = -1)

    o = tf.matmul(beta, hw_flatten(h))
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=input.shape)
    x = gamma * o + input

    return x