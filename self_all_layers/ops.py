import tensorflow as tf
import numpy as np
def pruned_weights(weights, prune_rate):
    '''
    weights.    : variables of tf
    pruned_rate : numpy array of shape [1, ]
    '''
    w_size = tf.reshape(weights, [-1]).shape[0]
    thre_idx = tf.cast(tf.multiply(tf.cast(w_size, 'float'), tf.subtract(1., prune_rate)), tf.int32)
    threshold = tf.nn.top_k(input = tf.abs(tf.reshape(weights, [-1])), k = w_size).values[thre_idx[0]]
    mask = tf.greater_equal(tf.abs(weights), threshold)
    mask = tf.cast(mask, 'float')
    W_ = tf.multiply(weights, mask)
    return W_
    
def trainable_variables(shape, name):
    '''
    shape : a list
    name  : a string
    '''
    initial = tf.truncated_normal(shape, stddev = 0.05)
    return tf.Variable(initial, name = name)
    
def conv_layers(inpt, kernel, bias, strides):
    '''
    inpt    : a tf.tensor
    kernel  : a tf.variable
    bias    : a tf.variable
    strides : a list
    '''
    return tf.nn.relu(tf.nn.conv2d(inpt, kernel, strides = strides, padding = 'SAME') + bias)
    
def scale_arrays(array):
    '''
    array : numpy array of shape [m, n]
    '''
    mean = np.mean(array, axis = 0)
    std = np.sqrt(np.mean(np.square(array - mean), axis = 0))
    std = np.where(std != 0., std, 0.001)
    return (array - mean) / std
    
def simple_scale_arrays(array):
    '''
    array : numpy array of shape [m, n]
    '''
    return array / array.max(0)
