# -*- coding:utf-8 -*-
"""
Load data.
Train and test CycleGAN.
"""
import os
import logging
from datetime import datetime
import time
import tensorflow as tf
from models import network_Gen, network_Dis
from utils import save_images

import sys
sys.path.append("../data")
from data import read_tfrecords
# load test image
import numpy as np
import cv2
import glob

class CycleGAN(object):
    def __init__(self, sess, tf_flags):
        self.sess = sess
        self.dtype = tf.float32
        self.net = tf_flags.which_net
        self.use_lsgan = tf_flags.use_lsgan

        self.output_dir = tf_flags.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"
        self.summary_dir = os.path.join(self.output_dir, "summary")

        self.is_training = (tf_flags.phase == "train")

        # data parameters
        self.image_w = 256
        self.image_h = 256
        self.image_c = 3

        self.real_X = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c])
        self.real_Y = tf.placeholder(self.dtype, [None, self.image_h, self.image_h, self.image_c])

        # network parameters. conv, num_filters.
        self.num_filters = 64

        # train
        if self.is_training:
            self.training_set = tf_flags.training_set
            self.sample_dir = "train_results"

            # Generator loss
            self.G_gen_loss = None
            self.lambda_G = tf_flags.lambda_G
            self.G_cycle_loss = None
            self.G_loss = None
            
            self.F_gen_loss = None
            self.lambda_F = tf_flags.lambda_G
            self.F_cycle_loss = None
            self.F_loss = None

            # Discriminator loss
            self.D_Y_real_loss = None
            self.D_Y_fake_loss = None
            self.D_Y_loss = None
            self.D_X_real_loss = None
            self.D_X_fake_loss = None
            self.D_X_loss = None

            # makedir aux dir
            self._make_aux_dirs()
            # compute and define loss
            self._build_training()
            # logging, only use in training
            log_file = self.output_dir + "/cyclegan.log"
            logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                                filename=log_file,
                                level=logging.DEBUG,
                                filemode='w')
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # test
            self.testing_set = tf_flags.testing_set
            # build model
            self.pred_fake_Y, self.pred_rec_X, self.pred_fake_X, self.pred_rec_Y = self._build_test()

    def _build_training(self):
        # Generator network. network_Gen() is Generator network. 
        # network_G: X --> Y'
        self.fake_Y = network_Gen(name='G', in_data=self.real_X, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=False)
        # network_F: Y --> X'
        self.fake_X = network_Gen(name='F', in_data=self.real_Y, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=False)
        # network_F: Y' --> X'
        self.rec_X = network_Gen(name='F', in_data=self.fake_Y, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=True)
        # network_G: X' --> Y'
        self.rec_Y = network_Gen(name='G', in_data=self.fake_X, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=True)
        
        # network variables
        # When training G and F jointly, the variables of network is not used.
        # self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        # self.F_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='F')

        # Discriminator network. network_Dis() is Discriminator network.
        # network_D_Y: judge X --> Y'
        self.pred_fake_Y = network_Dis(name='D_Y', in_data=self.fake_Y, num_filters=self.num_filters,
            is_training=self.is_training, reuse=False)
        # network_D_X: judge X --> Y'
        self.pred_fake_X = network_Dis(name='D_X', in_data=self.fake_X, num_filters=self.num_filters,
            is_training=self.is_training, reuse=False) 
        # input real_X
        self.pred_real_X = network_Dis(name='D_X', in_data=self.real_X, num_filters=self.num_filters,
            is_training=self.is_training, reuse=True)
        # input real_Y
        self.pred_real_Y = network_Dis(name='D_Y', in_data=self.real_Y, num_filters=self.num_filters,
            is_training=self.is_training, reuse=True)
        # network variables
        self.D_Y_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_Y')
        self.D_X_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_X')


        # G_gen_loss.
        if self.use_lsgan:
            self.G_gen_loss = tf.reduce_mean(tf.squared_difference(self.pred_fake_Y, tf.ones_like(self.pred_fake_Y)))
        else:
            self.G_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pred_fake_Y), 
                logits=self.pred_fake_Y))
        # G_cycle_loss.
        self.G_cycle_loss = tf.reduce_mean(tf.abs(self.rec_X - self.real_X)) * self.lambda_G # L1 loss.
        # G_loss
        self.G_loss = self.G_gen_loss + self.G_cycle_loss

        # F_gen_loss
        if self.use_lsgan:
            self.F_gen_loss = tf.reduce_mean(tf.squared_difference(self.pred_fake_X, tf.ones_like(self.pred_fake_X)))
        else:
            self.F_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pred_fake_X),
                logits=self.pred_fake_X))
        # F_cycle_loss
        self.F_cycle_loss = tf.reduce_mean(tf.abs(self.rec_Y - self.real_Y)) * self.lambda_F # L1 loss.
        # F_loss
        self.F_loss = self.F_gen_loss + self.F_cycle_loss

        # D_Y_loss
        if self.use_lsgan:
            self.D_Y_real_loss = tf.reduce_mean(
                tf.squared_difference(self.pred_real_Y, tf.ones_like(self.pred_real_Y)))
            self.D_Y_fake_loss = tf.reduce_mean(
                tf.squared_difference(self.pred_fake_Y, tf.zeros_like(self.pred_fake_Y)))
        else:
            self.D_Y_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.pred_real_Y), logits=self.pred_real_Y))
            self.D_Y_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.pred_fake_Y), logits=self.pred_fake_Y))
        
        self.D_Y_loss = (self.D_Y_real_loss + self.D_Y_fake_loss) * 0.5

        # D_X_loss
        if self.use_lsgan:
            self.D_X_real_loss = tf.reduce_mean(
                tf.squared_difference(self.pred_real_X, tf.ones_like(self.pred_real_X)))
            self.D_X_fake_loss = tf.reduce_mean(
                tf.squared_difference(self.pred_fake_X, tf.zeros_like(self.pred_fake_X)))
        else:
            self.D_X_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.pred_real_X), logits=self.pred_real_X))
            self.D_X_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.pred_fake_X), logits=self.pred_fake_X))
        
        self.D_X_loss = (self.D_X_real_loss + self.D_X_fake_loss) * 0.5

        # optimizer
        # Jointly train G and F, the var_list using default.
        self.Gen_loss = self.G_loss + self.F_loss
        self.Gen_opt = tf.train.AdamOptimizer().minimize(
            self.Gen_loss, name="Gen_opt")
        self.D_Y_opt = tf.train.AdamOptimizer().minimize(
            self.D_Y_loss, var_list=self.D_Y_variables, name="D_Y_opt")
        self.D_X_opt = tf.train.AdamOptimizer().minimize(
            self.D_X_loss, var_list=self.D_X_variables, name="D_X_opt")

        # summary
        tf.summary.scalar('G_gen_loss', self.G_gen_loss)
        tf.summary.scalar('G_cycle_loss', self.G_cycle_loss)
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('F_gen_loss', self.F_gen_loss)
        tf.summary.scalar('F_cycle_loss', self.F_cycle_loss)
        tf.summary.scalar('F_loss', self.F_loss)

        tf.summary.scalar('D_X_loss', self.D_X_loss)
        tf.summary.scalar('D_Y_loss', self.D_Y_loss)

        self.summary = tf.summary.merge_all()
        # summary and checkpoint
        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()


    def train(self, batch_size, training_steps, summary_steps, checkpoint_steps, save_steps):
        step_num = 0
        # restore last checkpoint
        latest_checkpoint = tf.train.latest_checkpoint("") 
        # use pretrained model, it can be self.checkpoint_dir, "", or you can appoint the saved checkpoint path.

        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.now(), 
                step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer()) # init all variables
            logging.info("{}: Init new training".format(datetime.now()))

        # Utilize TFRecord file to load data. change the tfrecords name for different datasets.
        # define class Read_TFRecords object.
        readerA = read_tfrecords.Read_TFRecords(filename=os.path.join(self.training_set, "summer_winter_trainA.tfrecords"), 
            batch_size=batch_size, image_h=self.image_h, image_w=self.image_w, image_c=self.image_c)
        readerB = read_tfrecords.Read_TFRecords(filename=os.path.join(self.training_set, "summer_winter_trainB.tfrecords"), 
            batch_size=batch_size, image_h=self.image_h, image_w=self.image_w, image_c=self.image_c)
        images_A = readerA.read()
        images_B = readerB.read()

        logging.info("{}: Done init data generators".format(datetime.now()))

        self.coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        try:
        # train
            c_time = time.time()
            for c_step in xrange(step_num + 1, training_steps + 1):
                batch_images_A, batch_images_B = self.sess.run([images_A, images_B])
                c_feed_dict = {
                    # TFRecord
                    self.real_X: batch_images_A,
                    self.real_Y: batch_images_B
                }
                # self.ops = [self.G_opt, self.F_opt, self.D_X_opt, self.D_Y_opt] # Don't generate the good results.
                self.ops = [self.Gen_opt, self.D_X_opt, self.D_Y_opt] # Jointly train
                self.sess.run(self.ops, feed_dict=c_feed_dict)

                # save summary
                if c_step % summary_steps == 0:
                    c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                    self.writer.add_summary(c_summary, c_step)

                    e_time = time.time() - c_time
                    time_periter = e_time / summary_steps
                    logging.info("{}: Iteration_{} ({:.4f}s/iter) {}".format(
                        datetime.now(), c_step, time_periter,
                        self._print_summary(c_summary)))
                    c_time = time.time() # update time

                # save checkpoint
                if c_step % checkpoint_steps == 0:
                    self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                        global_step=c_step)
                    logging.info("{}: Iteration_{} Saved checkpoint".format(
                        datetime.now(), c_step))

                if c_step % save_steps == 0:
                    self.X, self.X2Y_, self.Y_2X_, self.Y, self.Y2X_, self.X_2Y_ = self.sess.run(
                        [self.real_X, self.fake_Y, self.rec_X, self.real_Y, self.fake_X, self.rec_Y],
                        feed_dict=c_feed_dict)
                    save_images(self.X, self.X2Y_, self.Y_2X_, 
                        './{}/train_{}_{:04d}.png'.format(self.sample_dir, "X", c_step))
                    # X --> G --> Y' --> F --> X'.
                    save_images(self.Y, self.Y2X_, self.X_2Y_,
                        './{}/train_{}_{:04d}.png'.format(self.sample_dir, "Y", c_step))
                    # Y --> F --> X' --> G --> Y'. 
        except KeyboardInterrupt:
            print('Interrupted')
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            self.coord.request_stop()
            self.coord.join(threads)

        logging.info("{}: Done training".format(datetime.now()))

    def _build_test(self):
        # Generator network. network_Gen() is Generator network. 
        # network_G: X --> Y'
        self.fake_Y = network_Gen(name='G', in_data=self.real_X, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=False)
        # network_F: Y' --> X'
        self.rec_X = network_Gen(name='F', in_data=self.fake_Y, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=False)
        # network_F: Y --> X'
        self.fake_X = network_Gen(name='F', in_data=self.real_Y, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=True)
        # network_G: X' --> Y'
        self.rec_Y = network_Gen(name='G', in_data=self.fake_X, network=self.net, num_filters=self.num_filters,
            image_c=self.image_c, is_training=self.is_training, reuse=True)

        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name) 
        # define saver, after the network!

        return self.fake_Y, self.rec_X, self.fake_X, self.rec_Y

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def test(self):
        # Test only in a image.
        image_A_name = glob.glob(os.path.join(self.testing_set, "testA", "*.jpg"))
        image_B_name = glob.glob(os.path.join(self.testing_set, "testB", "*.jpg"))
        
        # In tensorflow, test image must divide 255.0.
        image_A = np.reshape(cv2.resize(cv2.imread(image_A_name[0], 1), 
            (self.image_h, self.image_w)), (1, self.image_h, self.image_w, self.image_c)) / 255.
        image_B = np.reshape(cv2.resize(cv2.imread(image_B_name[0], 1), 
            (self.image_h, self.image_w)), (1, self.image_h, self.image_w, self.image_c)) / 255.
        # OpenCV load image. the data format is BGR, w.t., (H, W, C). The default load is channel=3.

        print("{}: Done init data generators".format(datetime.now()))

        c_feed_dict = {
            self.real_X: image_A,
            self.real_Y: image_B
        }

        self.X2Y_, self.Y_2X_, self.Y2X_, self.X_2Y_ = self.sess.run(
            [self.fake_Y, self.rec_X, self.fake_X, self.rec_Y], feed_dict=c_feed_dict)

        return image_A, image_B, self.X2Y_, self.Y_2X_, self.Y2X_, self.X_2Y_ 

    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _print_summary(self, summary_string):
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result)