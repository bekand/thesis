# -*- coding: utf-8 -*-

import tensorflow as tf
import tfrecords_generator
import os.path as path

# --Constants--
DATA_DIR = path.expanduser('cnn_input')
NUM_CLASSES = 10
LABEL_NAMES = ['Art Nouveau (Modern)', 'Baroque', 'Expressionism',
               'Impressionism', 'Post-Impressionsim', 'Realism',
               'Rococo', 'Romanticism', 'Surrealism', 'Symbolism']

# tfrecords_generator.process_data_dir(DATA_DIR)
TRAIN_RECORD = (path.join(DATA_DIR, 'train\\train.tfrecords'))

def parser(example_proto):
    features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image': tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
    parsed_features['image'] = tf.reshape(parsed_features['image'], (224 ,224, 3))
    return parsed_features['label'], parsed_features['image']

def input_pipeline(path_to_recod, batch_size, parser=parser):
    dataset = tf.data.TFRecordDataset(path_to_recod)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(10 * batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    dataset = dataset.batch(batch_size)

    # Return an *initializable* iterator over the dataset, which will allow us to
    # re-initialize it at the beginning of each epoch.
    return dataset.make_initializable_iterator()

with tf.Session() as sess:
    iterator = input_pipeline(TRAIN_RECORD, 10)
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    labels, images = sess.run(iterator.get_next())
    sess.run(iterator.initializer)
    labels, images = sess.run(iterator.get_next())
    print(labels)

    tf.get_default_graph().finalize()

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coordinator)

    # Wait for threads to finish, cleanup
    coordinator.request_stop()
    coordinator.join(threads)
    sess.close()