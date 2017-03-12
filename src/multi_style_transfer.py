import scipy.io
import numpy as np
import src.vgg19 as vgg

from src.op import op
from src.layers import *
from src.functions import *


class mst(op):
    def __init__(self,args, sess):
        op.__init__(self, args, sess)

    def mst_net(self, x, style_control=None, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            b,h,w,c = x.get_shape().as_list()

            x = conv_layer(x, 32, 9, 1, style_control=style_control, name='conv1')
            x = conv_layer(x, 64, 3, 2, style_control=style_control, name='conv2')
            x = conv_layer(x, 128, 3, 2, style_control=style_control, name='conv3')
            x = residual_block(x, 3, style_control=style_control, name='res1')
            x = residual_block(x, 3, style_control=style_control, name='res2')
            x = residual_block(x, 3, style_control=style_control, name='res3')
            x = residual_block(x, 3, style_control=style_control, name='res4')
            x = residual_block(x, 3, style_control=style_control, name='res5')
            x = conv_tranpose_layer(x, 64, 3, 2, style_control=style_control, name='up_conv1')
            x = pooling(x)
            x = conv_tranpose_layer(x, 32, 3, 2, style_control=style_control, name='up_conv2')
            x = pooling(x)
            x = conv_layer(x, 3, 9, 1, relu=False, style_control=style_control,  name='output')
            preds = tf.nn.tanh(x) * 150 + 255./2
        return preds


    def build_model(self):
        # Set content / style input
        # content_input
        b = self.batch_size; h = self.content_data_size; w = self.content_data_size;
        self.content_input = tf.placeholder(tf.float32, shape=[b, h, w, 3], name='content_input')

        # style_input
        style_img = get_image(self.style_image)
        style_idx = [i for i, x in enumerate(self.style_control) if not x == 0][0]
        print 'style_idx : {}'.format(style_idx)
        style_input = tf.constant((style_img[np.newaxis, ...]), dtype=tf.float32)

        # MST_output (Pastiche)
        MST_output    = self.mst_net(self.content_input, style_control=self.style_control)

        # VGG network
        weights = scipy.io.loadmat('src/vgg19.mat')
        vgg_mean = tf.constant(np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3)), dtype='float32')

        content_feats = vgg.net(self.content_input - vgg_mean, weights)
        style_feats = vgg.net(style_input - vgg_mean, weights)
        MST_feats = vgg.net(MST_output - vgg_mean, weights)

        c_loss = self.content_weights * euclidean_loss(MST_feats[-1], content_feats[-1])
        s_loss = self.style_weights * sum([style_loss(MST_feats[i], style_feats[i]) for i in range(5)])
        tv_loss = self.tv_weight * total_variation(MST_output)
        loss = c_loss + s_loss + tv_loss

        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if '{0}'.format(style_idx) + '_style' in var.name]

        if style_idx == 0:
            train_opt = tf.train.AdamOptimizer(self.learning_rate, self.momentum).minimize(loss)
        else:
            train_opt = tf.train.AdamOptimizer(self.learning_rate, self.momentum).minimize(loss, var_list=vars)

        self.optimize = [train_opt, loss, c_loss, s_loss, tv_loss]
        self.saver = tf.train.Saver(var_list=t_vars, max_to_keep=(self.max_to_keep))
        self.sess.run(tf.global_variables_initializer())


    def train(self,Train_flag):
        op.train(self,Train_flag)

    def test(self, Train_flag):
        op.test(self,Train_flag)

    def save(self):
        op.save(self)

    def load(self):
        op.load(self)
