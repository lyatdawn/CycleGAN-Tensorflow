# -*- coding:utf-8 -*-
"""
Generate TFRecords file, for training.
"""
import os
import glob
import tensorflow as tf

if __name__ == '__main__':
    # change data_root for different datasets, e.g. vangogh2photo.
    data_root = "../datasets/summer2winter"
    for filename in ["trainA", "trainB"]:
        # For example, trainA is apple; trainB is orange.
        data_dir = os.path.join(data_root, filename) # e.g., ../datasets/apple2orange/trainA.

        # TFRecordWriter, dump to tfrecords file
        # TFRecord file name, change save_name for different datasets, e.g. vangogh_photo_.
        save_name = "summer_winter_" + filename
        writer = tf.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords", save_name + ".tfrecords"))
        for image_file in glob.glob(os.path.join(data_dir, "*.jpg")):
            # The first method to load image.
            '''
            image_raw = Image.open(image_file) # It image is RGB, then mode=RGB; otherwise, mode=L.
            # reszie image. In reading the TFRecords file, if you want resize the image, you could put image 
            # height and width into the Feature.
            # In this way, when reading the TFRecords file, it can use width and height.
            width = image_raw.size[0]
            height = image_raw.size[1]
            # put image height and width into the Feature.
            "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
            "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))

            image_raw = image_raw.tobytes()
            # Transform image to byte.
            '''

            # Second method to load image.
            image_raw = tf.gfile.FastGFile(image_file, 'rb').read() # image data type is string. read and binary.

            # write bytes to Example proto buffer.
            example = tf.train.Example(features=tf.train.Features(feature={
                "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))
            
            writer.write(example.SerializeToString()) # Serialize To String
        
        writer.close()
