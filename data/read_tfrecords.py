# -*- coding:utf-8 -*-
"""
Read TFRecords file.
"""
import os
import tensorflow as tf
import numpy as np
import scipy.misc

class Read_TFRecords(object):
    def __init__(self, filename, batch_size=64,
        image_h=256, image_w=256, image_c=3, num_threads=8, capacity_factor=3, min_after_dequeue=1000):
        '''
        filename: TFRecords file path.
        num_threads: TFRecords file load thread.
        capacity_factor: capacity.
        '''
        self.filename = filename
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.num_threads = num_threads
        self.capacity_factor = capacity_factor
        self.min_after_dequeue = min_after_dequeue

    def read(self):
        # read a TFRecords file, return tf.train.batch/tf.train.shuffle_batch object.
        reader = tf.TFRecordReader()
        
        filename_queue = tf.train.string_input_producer([self.filename])
        key, serialized_example = reader.read(filename_queue)
        
        features = tf.parse_single_example(serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([], tf.string),
            })
       
        image = tf.image.decode_jpeg(features["image_raw"], channels=self.image_c, name="decode_image")
        # not need Crop and other random augmentations.
        # image resize and transform type.
        # Utilize tf.gfile.FastGFile() to generate TFRecords file, in this way, it could use resize_images directly.
        if self.image_h is not None and self.image_w is not None:
            image = tf.image.resize_images(image, [self.image_h, self.image_w], 
                method=tf.image.ResizeMethod.BICUBIC)
        image = tf.cast(image, tf.float32) / 255.0 # convert to float32

        # tf.train.batch/tf.train.shuffle_batch object.
        # Using asynchronous queues
        batch_data = tf.train.shuffle_batch([image],
            batch_size=self.batch_size,
            capacity=self.min_after_dequeue + self.capacity_factor * self.batch_size,
            min_after_dequeue=self.min_after_dequeue,
            num_threads=self.num_threads,
            name='images')
        
        return batch_data # return list or dictionary of tensors.


if __name__ == '__main__':
    # Test class Read_TFRecords.
    # define class Read_TFRecords object.
    data_dir = "../datasets/tfrecords"
    readerA = Read_TFRecords(filename=os.path.join(data_dir, "trainA.tfrecords"), batch_size=64,
        image_h=256, image_w=256, image_c=3)
    readerB = Read_TFRecords(filename=os.path.join(data_dir, "trainB.tfrecords"), batch_size=64,
        image_h=256, image_w=256, image_c=3)
    images_A = readerA.read()
    images_B = readerB.read()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        for i in range(2):
        # while not coord.should_stop():
            # Utilize for or while loop to load data, the data type is Tensor. Use sess.run() to transform numpy array.
            batch_images_A, batch_images_B = sess.run([images_A, images_B])
            print("image shape A: {}".format(batch_images_A.shape))
            print("image shape B: {}".format(batch_images_B.shape))

            # save
            scipy.misc.toimage(batch_images_A[0, :, :, :], cmin=0., cmax=1.).save("A.png")
            scipy.misc.toimage(batch_images_B[0, :, :, :], cmin=0., cmax=1.).save("B.png")

            step += 1
            print(step)
    except KeyboardInterrupt:
        print('Interrupted')
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
