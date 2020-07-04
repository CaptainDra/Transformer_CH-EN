# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function
import tensorflow as tf

#归一化
def normalize(inputs, epsilon = 1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta 
    return outputs

#嵌入层
def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True,scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32, shape=[vocab_size, num_units], initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5)      
    return outputs
    
#隐藏编码层
def positional_encoding(inputs, num_units,zero_pad=True, scale=True, scope="positional_encoding", reuse=None):

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  
        lookup_table = tf.convert_to_tensor(position_enc)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs

#平滑
def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
    
    

            
