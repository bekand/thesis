# Based on:
# https://github.com/damienpontifex/BlogCodeSamples/blob/master/DataToTfRecords/directories-to-tfrecords.py

import os.path as path
import os
import glob
import tensorflow as tf
import pickle
import numpy as np
import sys
import random
from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

 # Create a Int64List Feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a BytesList Feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(dataset_name,
                        data_directory,
                        class_map,
                        directories_as_labels=True,
                        files='**/*.jpg'):

    # Create a dataset of file path and class tuples for each file
    filenames = glob.glob(path.join(data_directory, files))
    classes = (path.basename(path.dirname(name)) for name in filenames) if directories_as_labels else [None] * len(filenames)
    dataset = list(zip(filenames, classes))
    random.shuffle(dataset)

    num_examples = len(filenames)
    record_filename = path.join(data_directory, dataset_name+'.tfrecords')

    with tf.python_io.TFRecordWriter(record_filename) as writer:
        print('Writing'+record_filename +'\n')
        for index, sample in enumerate(dataset):
            file_path, label = sample
            print('Processing: '+ file_path +'\n')
            image = Image.open(file_path)
            image = image.resize((224, 224))
            image_raw = np.array(image).tostring()

            features = {
                'label': _int64_feature(class_map[label]),
                'text_label': _bytes_feature(label.encode()),
                'image': _bytes_feature(image_raw)
            }
            sys.stdout.flush()
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

def process_data_dir(data_dir: str):

    train_data_dir = path.join(data_dir, 'train')
    test_data_dir = path.join(data_dir, 'test')
    # Get names of classes based on folder names
    class_names = os.listdir(train_data_dir)
    # Map class names to integer labels
    class_name2id = {label: index for index, label in enumerate(class_names)}

    # Persist this mapping so it can be loaded when training for decoding
    with open(path.join(data_dir, 'class_name2id.p'), 'wb') as p:
        pickle.dump(class_name2id, p, protocol=pickle.HIGHEST_PROTOCOL)

    convert_to_tfrecord('train', train_data_dir, class_name2id)
    # convert_to_tfrecord('test', test_data_dir, class_name2id, directories_as_labels=False)
