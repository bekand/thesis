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
import cv2


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

    # Create a dataset of file path and class tuples for each file, then shuffle it
    filenames = glob.glob(path.join(data_directory, files))
    classes = (path.basename(path.dirname(name)) for name in filenames)
    dataset = list(zip(filenames, classes))
    random.shuffle(dataset)

    record_filename = path.join(data_directory, dataset_name + '.tfrecords')

    # Create txt to write skipped files to
    with tf.python_io.TFRecordWriter(record_filename) as writer, open('badfiles' + dataset_name + '.txt', 'w+') as badf:
        for index, sample in enumerate(dataset):
            file_path, label = sample
            image = cv2.imread(file_path)

            # Try to resize image, then write features (image and label) to the tfrecord
            # If cv2 can't process it, or the resulting shape is not (224, 224, 3), skip
            try:
                image = cv2.resize(image, (224, 224))
                if np.shape(image) == (224, 224, 3):
                    print('Processing: ' + file_path + '               ', end='\r')
                    image_raw = np.array(image).tostring()
                    features = {
                        'label': _int64_feature(class_map[label]),
                        'image': _bytes_feature(image_raw)
                    }
                    sys.stdout.flush()
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
                else:
                    err = 'cannot process, wrong shape: ' + file_path + '\n'
                    print(err)
                    badf.write(err)
            except:
                err = 'cannot process, exception: ' + file_path + '\n'
                print(err)
                print(sys.exc_info()[0])
                badf.write(err)
            sys.stdout.flush()


def main(data_dir, train_dir='train', test_dir='test'):
    train_data_dir = path.join(data_dir, train_dir)
    test_data_dir = path.join(data_dir, test_dir)
    if (path.exists(data_dir) and path.exists(train_data_dir) and path.exists(test_data_dir)):
        # Get names of classes based on folder names
        class_names = os.listdir(train_data_dir)
        # Map class names to integer labels
        class_name2id = {label: index for index, label in enumerate(class_names)}

        # Persist this mapping so it can be loaded when training for decoding
        with open(path.join(data_dir, 'class_name2id.p'), 'wb') as p:
            pickle.dump(class_name2id, p, protocol=pickle.HIGHEST_PROTOCOL)

        convert_to_tfrecord(train_dir, train_data_dir, class_name2id)
        convert_to_tfrecord(test_dir, test_data_dir, class_name2id)
    else:
        print('Error: The given directory does not exist.')


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except(IndexError):
        print("No directory given.")
