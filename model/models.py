# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
import tensorflow as tf
import utils

# In style transform, use Instance Norm, instead of Batch Norm.
def network_Gen(name, in_data, network, num_filters, image_c, is_training, n_blocks=6, reuse=False):
    # num_filters=64; image_c=3.
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        if network == "unet":
            # TODO: utilize unet to transform images.
            pass
        elif network == "resnet":
            # In tf, utlize tf.pad() to pad tensor. It will be used before conv.
            # In conv, if padding='VALID', it will use tf.pad() to pad tensor.
            c_out_res_g1 = utils.res_mod_layers(in_data=in_data, num_filters=num_filters, 
                kernel_size=7, strides=[1, 1], padding='VALID', is_training=is_training, 
                relu="relu", alpha=0.2, ReflectionPadding=True, padding_size=[[0,0],[3,3],[3,3],[0,0]])
            
            c_in_G = c_out_res_g1
            n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2**i
                c_out_res_g2 = utils.res_mod_layers(in_data=c_in_G, num_filters=num_filters * mult * 2, 
                    kernel_size=3, strides=[2, 2], padding="SAME", is_training=is_training, 
                    relu="relu", alpha=0.2, ReflectionPadding=False)
                c_in_G = c_out_res_g2

            mult = 2**n_downsampling
            for i in range(n_blocks):
                c_out_res_g3 = utils.res_block(in_data=c_in_G, num_filters=num_filters * mult, 
                    kernel_size=3, strides=[1, 1], padding="VALID", is_training=is_training,
                    relu="relu", alpha=0.2, ReflectionPadding=True, padding_size=[[0,0],[1,1],[1,1],[0,0]])
                c_in_G = c_out_res_g3

            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                c_out_gdeconv = tf.layers.conv2d_transpose(
                    inputs=c_in_G,
                    filters=int(num_filters * mult / 2),
                    kernel_size=3,
                    strides=[2, 2],
                    padding="SAME")
                # Instance Norm
                c_out_gbn = tf.contrib.layers.instance_norm(
                    inputs=c_out_gdeconv,
                    center=True,
                    scale=True,
                    epsilon=1e-08)
                c_out_grelu = tf.nn.relu(c_out_gbn)
                c_in_G = c_out_grelu

            c_out_gpad = tf.pad(c_in_G, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
            c_out_gconv = tf.layers.conv2d(
                inputs=c_out_gpad,
                filters=image_c,
                kernel_size=7,
                strides=[1, 1],
                padding="VALID")
            c_out_gtanh = tf.nn.tanh(c_out_gconv)

    return c_out_gtanh
            
def network_Dis(name, in_data, num_filters, is_training, use_sigmoid=False, n_layers=3, reuse=False):
    # num_filters=64; image_c=3.
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        c_out_dconv = tf.layers.conv2d(
            inputs=in_data,
            filters=num_filters,
            kernel_size=4,
            strides=[2, 2],
            padding="SAME")
        c_out_drelu = tf.nn.leaky_relu(features=c_out_dconv, alpha=0.2)

        c_in_D = c_out_drelu
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            c_out_res_d1 = utils.res_mod_layers(in_data=c_in_D, num_filters=num_filters * nf_mult, 
                kernel_size=4, strides=[2, 2], padding="SAME", is_training=is_training, 
                relu="leaky_relu", alpha=0.2, ReflectionPadding=False)
            c_in_D = c_out_res_d1

        nf_mult = min(2**n_layers, 8)
        c_out_res_d2 = utils.res_mod_layers(in_data=c_in_D, num_filters=num_filters * nf_mult, 
            kernel_size=4, strides=[1, 1], padding="SAME", is_training=is_training,
            relu="leaky_relu", alpha=0.2, ReflectionPadding=False)

        c_out_dconv = tf.layers.conv2d(
            inputs=c_out_res_d2,
            filters=1,
            kernel_size=4,
            strides=[1, 1],
            padding="SAME")

        if use_sigmoid:
            c_out_dsigmoid = tf.nn.sigmoid(c_out_dconv)
            return c_out_dsigmoid
        else:
            return c_out_dconv