import tensorflow as tf


def conv_layer(net, num_filters, filter_size, strides, style_control=None, relu=True, name='conv'):
    with tf.variable_scope(name):
        b,w,h,c = net.get_shape().as_list()
        weights_shape = [filter_size, filter_size, c, num_filters]
        weights_init = tf.get_variable(name, shape=weights_shape, initializer=tf.truncated_normal_initializer(stddev=.01))
        strides_shape = [1, strides, strides, 1]

        p = (filter_size - 1) / 2
        if strides == 1:
            net = tf.pad(net, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding="VALID")
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding="SAME")

        net = conditional_instance_norm(net, style_control=style_control)
        if relu:
            net = tf.nn.relu(net)

    return net


def conv_tranpose_layer(net, num_filters, filter_size, strides, style_control=None, name='conv_t'):
    with tf.variable_scope(name):
        b, w, h, c = net.get_shape().as_list()
        weights_shape = [filter_size, filter_size, num_filters, c]
        weights_init = tf.get_variable(name, shape=weights_shape, initializer=tf.truncated_normal_initializer(stddev=.01))

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        p = (filter_size - 1) / 2
        if strides == 1:
            net = tf.pad(net, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding="VALID")
        else:
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding="SAME")
        net = conditional_instance_norm(net, style_control=style_control)

    return tf.nn.relu(net)


def residual_block(net, filter_size=3, style_control=None, name='res'):
    with tf.variable_scope(name+'_a'):
        tmp = conv_layer(net, 128, filter_size, 1, style_control=style_control)
    with tf.variable_scope(name+'_b'):
        output = net + conv_layer(tmp, 128, filter_size, 1, style_control=style_control, relu=False)
    return output


def conditional_instance_norm(net, style_control=None, name='cond_in'):
    with tf.variable_scope(name):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)

        var_shape = [channels]
        shift = []
        scale = []

        for i in range(len(style_control)):
            with tf.variable_scope('{0}'.format(i) + '_style'):
                shift.append(tf.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.)))
                scale.append(tf.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.)))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

        idx = [i for i, x in enumerate(style_control) if not x == 0]

        style_scale = reduce(tf.add, [scale[i]*style_control[i] for i in idx]) / sum(style_control)
        style_shift = reduce(tf.add, [shift[i]*style_control[i] for i in idx]) / sum(style_control)
        output = style_scale * normalized + style_shift

    return output


def instance_norm(net, train=True, name='in'):
    with tf.variable_scope(name):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
        scale = tf.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def pooling(input):
    return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')


def total_variation(preds):
     # total variation denoising
     b,w,h,c = preds.get_shape().as_list()
     y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:w-1,:,:])
     x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:h-1,:])
     tv_loss = 2*(x_tv + y_tv)/b/w/h/c
     return tv_loss


def euclidean_loss(input_, target_):
    b,w,h,c = input_.get_shape().as_list()
    return 2 * tf.nn.l2_loss(input_- target_) / b/w/h/c


def gram_matrix(net):
    b,h,w,c = net.get_shape().as_list()
    feats = tf.reshape(net, (b, h*w, c))
    feats_T = tf.transpose(feats, perm=[0,2,1])
    grams = tf.matmul(feats_T, feats) / h/w/c
    return grams


def style_loss(input_, style_):
    b,h,w,c = input_.get_shape().as_list()
    input_gram = gram_matrix(input_)
    style_gram = gram_matrix(style_)
    return 2 * tf.nn.l2_loss(input_gram - style_gram)/b/c/c

