# Ref: https://github.com/arashsaber/Deep-Convolutional-AutoEncoder/blob/master/ConvolutionalAutoEncoder.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#   ---------------------------------
def conv2d(input, name, kshape, strides=[1, 1, 1, 1], pad='SAME'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.glorot_normal_initializer())
        out = tf.nn.conv2d(input,W,strides=strides, padding=pad)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=(1, 1), pad='SAME'):
    with tf.variable_scope(name):
        in_channels = int(input.shape[-1])
        W = tf.get_variable(name='w_'+name,
                            shape=[kshape[0], kshape[1], n_outputs, in_channels],
                            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(name='b_'+name,
                            shape=[n_outputs],
                            initializer=tf.glorot_normal_initializer())
        in_shape = tf.shape(input)
        out_h = in_shape[1] * strides[0]
        out_w = in_shape[2] * strides[1]
        output_shape = tf.stack([in_shape[0], out_h, out_w, n_outputs])
        strides_4d = [1, strides[0], strides[1], 1]
        out = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
                                     strides=strides_4d, padding=pad)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
#   ---------------------------------
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape,
                             strides=strides,
                             padding='SAME')
        return out
#   ---------------------------------
def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(name):
        out = tf.image.resize(input, size=size, method='bilinear')
        return out
#   ---------------------------------
def fullyConnected(input, name, output_size):
    with tf.variable_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.glorot_normal_initializer())
        input = tf.reshape(input, [-1, input_size])
        out = tf.add(tf.matmul(input, W), b)
        return out
#   ---------------------------------
def dropout(input, name, keep_rate):
    with tf.variable_scope(name):
        out = tf.nn.dropout(input, rate=1.0 - keep_rate)
        return out
#   ---------------------------------
