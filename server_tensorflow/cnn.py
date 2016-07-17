

import tensorflow as tf

def conv_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
               bias_term=True, weights_initializer=None, biases_initializer=None):
    # input has shape [batch, in_height, in_width, in_channels]
    input_dim = bottom.get_shape().as_list()[-1]

    # weights and biases variables
    with tf.variable_scope(name):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.random_normal_initializer()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # filter has shape [filter_height, filter_width, in_channels, out_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, input_dim, output_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)

    conv = tf.nn.conv2d(bottom, filter=weights,
        strides=[1, stride, stride, 1], padding=padding)
    if bias_term:
        conv = tf.nn.bias_add(conv, biases)
    return conv

def conv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                    bias_term=True, weights_initializer=None, biases_initializer=None):
    conv = conv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                      bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(conv)
    return relu

def deconv_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                 bias_term=True, weights_initializer=None, biases_initializer=None):
    # input_shape is [batch, in_height, in_width, in_channels]
    input_shape = bottom.get_shape().as_list()
    batch_size, input_height, input_width, input_dim = input_shape
    output_shape = [batch_size, input_height*stride, input_width*stride, output_dim]

    # weights and biases variables
    with tf.variable_scope(name):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.random_normal_initializer()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # filter has shape [filter_height, filter_width, out_channels, in_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, output_dim, input_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)

    deconv = tf.nn.conv2d_transpose(bottom, filter=weights,
        output_shape=output_shape, strides=[1, stride, stride, 1],
        padding=padding)
    if bias_term:
        deconv = tf.nn.bias_add(deconv, biases)
    return deconv

def deconv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                      bias_term=True, weights_initializer=None, biases_initializer=None):
    deconv = deconv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                          bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(deconv)
    return relu

def pooling_layer(name, bottom, kernel_size, stride):
    pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool

def fc_layer(name, bottom, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    flat_bottom = tf.reshape(bottom, [-1, input_dim])

    # weights and biases variables
    with tf.variable_scope(name):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.random_normal_initializer()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # weights has shape [input_dim, output_dim]
        weights = tf.get_variable("weights", [input_dim, output_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)
    if bias_term:
        fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
    else:
        fc = tf.matmul(flat_bottom, weights)
    return fc

def fc_relu_layer(name, bottom, output_dim, bias_term=True,
                  weights_initializer=None, biases_initializer=None):
    fc = fc_layer(name, bottom, output_dim, bias_term, weights_initializer,
                  biases_initializer)
    relu = tf.nn.relu(fc)
    return relu

def batch_norm_layer(name, bottom, phase_train, include_scale=True,
        include_bias=True, exp_avg_decay=0.99):
    shape = bottom.get_shape().as_list()
    channels = shape[-1]

    # Convert 2D bottom to N x H x W x C format
    assert(len(shape) == 2 or len(shape) == 4)
    if len(shape) == 2:
        bottom = tf.reshape(bottom, [-1, 1, 1, channels])

    with tf.variable_scope(name):
        # Exponential moving average of mean and variance
        mean_ema = tf.get_variable("mean_ema", trainable=False,
            initializer=tf.zeros_initializer([1, 1, 1, channels]))
        variance_ema = tf.get_variable("variance_ema", trainable=False,
            initializer=tf.zeros_initializer([1, 1, 1, channels]))

        scales, biases = None, None
        if include_scale:
            scales = tf.get_variable("scales", [1, 1, 1, channels])
        if include_bias:
            biases = tf.get_variable("biases", [1, 1, 1, channels])

        eps = 1e-6
        decay = tf.constant(exp_avg_decay, dtype=tf.float32)
        if phase_train:
            # Estimate mean and variance from current batch
            mean, variance = tf.nn.moments(bottom, [0, 1, 2], keep_dims=True)

            # Operations to update mean and variance with moving average:
            #   a_new = decay * a + (1-decay) * x
            #         = a - (1-decay) * a + (1-decay) * x
            #         = a + (1-decay) * (x - a)
            update_mean = tf.assign_add(mean_ema,
                (1 - decay) * (mean - mean_ema))
            update_variance = tf.assign_add(variance_ema,
                (1 - decay) * (variance - variance_ema))
            # control_dependencies ensures that update_mean and update_variance
            # will be executed during training
            with tf.control_dependencies([update_mean, update_variance]):
                normalized = tf.nn.batch_normalization(bottom, mean, variance,
                    biases, scales, eps)
        else:
            normalized = tf.nn.batch_normalization(bottom, mean_ema, variance_ema,
                biases, scales, eps)

    # Reshape back
    if len(shape) == 2:
        normalized = tf.reshape(normalized, shape)
    return normalized
