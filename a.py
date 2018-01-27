# -*- coding: utf-8 -*-

import tensorflow as tf
# from tqdm import tqdm
# import cv2

# --Constants--
TRAIN_DIR = '../cnn_input/train/'
TEST_DIR = '../cnn_input/test/'

NUM_CLASSES = 10
LABEL_NAMES = ['Art Nouveau', 'Baroque', 'Expressionism',
               'Impressionism', 'Post-Impressionsim', 'Realism',
               'Rococco', 'Romanticism', 'Surrealism', 'Symbolism']

# --Helper Functions --
def label_as_one_hot(label):
    try:
        label = label.decode()
    except AttributeError:
        pass
    return tf.one_hot(indices=LABEL_NAMES.index(label), depth=NUM_CLASSES)

# A TensorFlow Session for use in interactive contexts, such as a shell.
sess = tf.InteractiveSession()

csv_path = tf.train.string_input_producer([TRAIN_DIR+'train.csv'])
reader = tf.TextLineReader()
_, csv_content = reader.read(csv_path)

# defaults, also gives the shape of the output tensors for tf.decode
record_defaults = [[""], [""], [""], [""]]
artist, style, title, filename = tf.decode_csv(csv_content, record_defaults)

sess.run(tf.global_variables_initializer())
# tf.get_default_graph().finalize()

# #
coordinator = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coordinator)
print(label_as_one_hot(style.eval()).eval())

# Wait for threads to finish, cleanup
coordinator.request_stop()
coordinator.join(threads)
sess.close()
