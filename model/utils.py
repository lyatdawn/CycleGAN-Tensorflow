# -*- coding:utf-8 -*-
"""
Util module.
"""
import tensorflow as tf
import numpy as np
import scipy.misc

def save_images(label, G_output, F_output, image_path):
    image = np.concatenate([label, G_output, F_output], axis=2) # concat 4D array, along width.
    max_samples = int(image.shape[0]) # batch_size == 4.
    image = image[0:max_samples, :, :, :]
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)
    # concat 3D array, along axis=0, w.t. along height. shape: (1024, 256, 3).

    # save image.
    # scipy.misc.toimage(), array is 2D(gray, reshape to (H, W)) or 3D(RGB).
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(image_path) # image_path contain image path and name.

def res_mod_layers(in_data, num_filters, kernel_size, strides, padding,
        is_training, relu, alpha, ReflectionPadding, padding_size=None):
    if ReflectionPadding:
        input_data = tf.pad(in_data, padding_size, 'REFLECT')
    else:
        input_data = in_data
    # conv
    conv_out = tf.layers.conv2d(
        inputs=input_data,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
    # Instance Norm
    bn_out = tf.contrib.layers.instance_norm(
        inputs=conv_out,
        center=True,
        scale=True,
        epsilon=1e-08)
    # relu
    if relu == "relu":
        act_out = tf.nn.relu(bn_out)
    if relu == "leaky_relu":
        act_out = tf.nn.leaky_relu(bn_out, alpha=alpha)

    return act_out

def res_block(in_data, num_filters, kernel_size, strides, padding,
        is_training, relu, alpha, ReflectionPadding, padding_size):
    # first conv + bn + relu
    res_out = res_mod_layers(in_data=in_data, num_filters=num_filters, 
        kernel_size=kernel_size, strides=strides, padding=padding, is_training=is_training, 
        relu=relu, alpha=alpha, ReflectionPadding=ReflectionPadding, padding_size=padding_size)
    # conv + bn
    padding_out = tf.pad(res_out, padding_size, 'REFLECT')
    # conv
    resblock_conv_out = tf.layers.conv2d(
        inputs=padding_out,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
    # Instance Norm
    resblock_bn_out = tf.contrib.layers.instance_norm(
        inputs=resblock_conv_out,
        center=True,
        scale=True,
        epsilon=1e-08)
    # output
    output = in_data + resblock_bn_out

    return output


def unet_block(in_data, num_filters, kernel_size, strides, padding, is_training):
    # TODO: utilize unet to transform images.
    pass