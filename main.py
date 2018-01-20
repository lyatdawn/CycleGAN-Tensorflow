# -*- coding:utf-8 -*-
"""
An implementation of CycleGan using TensorFlow (work in progress).
"""
import tensorflow as tf
import numpy as np
from model import cyclegan
import cv2
import scipy.misc # save image


def main(_):
    tf_flags = tf.app.flags.FLAGS
    # gpu config.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess: 
        # when use queue to load data, not use with to define sess
            train_model = cyclegan.CycleGAN(sess, tf_flags)
            train_model.train(tf_flags.batch_size, tf_flags.training_steps, 
                              tf_flags.summary_steps, tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = cyclegan.CycleGAN(sess, tf_flags)
            test_model.load(tf_flags.checkpoint)
            test_A, test_B, X2Y_, Y_2X_, Y2X_, X_2Y_ = test_model.test()
            # return numpy ndarray.
            
            # save two images.
            filename_A2B = "A2B.png"
            filename_B2A = "B2A.png"
            image_A2B = np.concatenate([test_A[0,:], X2Y_[0,:], Y_2X_[0,:]], axis=1)
            image_B2A = np.concatenate([test_B[0,:], Y2X_[0,:], X_2Y_[0,:]], axis=1)

            scipy.misc.toimage(image_A2B, cmin=0., cmax=1.).save(filename_A2B) # image_path contain image path and name.
            scipy.misc.toimage(image_B2A, cmin=0., cmax=1.).save(filename_B2A) # image_path contain image path and name.
            # cv2.imwrite(filename_A2B, np.uint8(image_A2B.clip(0., 1.) * 255.))
            # cv2.imwrite(filename_B2A, np.uint8(image_B2A.clip(0., 1.) * 255.))

            # Utilize cv2.imwrite() to save images.
            print("Saved files: {}, {}".format(filename_A2B, filename_B2A))

if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output", 
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train", 
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("which_net", "resnet", 
                               "which network will use: unet/resnet.")
    tf.app.flags.DEFINE_bool("use_lsgan", True, 
                             "which loss will use: MSE/Sigmoid Cross Entropy.")
    tf.app.flags.DEFINE_float("lambda_G", 10., 
                              "scale G_clcye_loss.")
    tf.app.flags.DEFINE_float("lambda_F", 10., 
                              "scale F_clcye_loss.")
    tf.app.flags.DEFINE_string("training_set", "./datasets/tfrecords", 
                               "dataset path for training.")
    tf.app.flags.DEFINE_string("testing_set", "./datasets/test/apple2orange", 
                               "dataset path for testing one image pair.")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_steps", 100000, 
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100, 
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 500, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None, 
                                "checkpoint name for restoring.")
    tf.app.run(main=main)
