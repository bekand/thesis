import os.path as path
import tensorflow as tf
from tensorflow import keras as K
import random


# --Constants--
NUM_CLASSES = 10
LABEL_NAMES = ['Art Nouveau (Modern)', 'Baroque', 'Expressionism',
               'Impressionism', 'Post-Impressionsim', 'Realism',
               'Rococo', 'Romanticism', 'Surrealism', 'Symbolism']
INPUT_SHAPE = (224, 224, 3)
EPOCHS = 100
BATCH_SIZE = 16
if path.isdir("/painting-styles"):  # we are on floydhub
    TRAIN_RECORD = '/painting-styles/train.tfrecords'
    TEST_RECORD = '/painting-styles/test.tfrecords'
else:
    DATA_DIR = path.expanduser('cnn_input')
    TRAIN_RECORD = (path.join(DATA_DIR, 'painting-styles\\train.tfrecords'))
    TRAIN_RECORD = (path.join(DATA_DIR, 'painting-styles\\test.tfrecords'))


# --Utility functions--
# convert the tfrecord's elements back to usable format
def parser(example_proto):
    features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image': tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
    parsed_features['image'] = tf.reshape(parsed_features['image'], INPUT_SHAPE)
    return tf.one_hot(parsed_features['label'], NUM_CLASSES), parsed_features['image']


# randomly alter this batch of images (flip/darken/brighten)
def augmenter(label, image):
    flip_chance = random.randint(0, 100)
    rand_brightness_chance = random.randint(0, 100)

    if flip_chance <= 20:
        image = tf.image.flip_left_right(image)
    if rand_brightness_chance <= 40:
        image = tf.image.random_brightness(image, 0.4)
    return label, image


def input_pipeline(path_to_recod, batch_size=BATCH_SIZE, parser_function=parser):
    dataset = tf.data.TFRecordDataset(path_to_recod)
    dataset = (dataset.repeat()
               .shuffle(1000)
               .map(parser_function)
               .map(augmenter)
               .batch(batch_size)
               .prefetch(batch_size))

    return dataset


pre_trained = K.applications.vgg16.VGG16(include_top=False,
                                         weights='imagenet',
                                         input_shape=INPUT_SHAPE)

# Freeze the layers except the last 4 layers
for layer in pre_trained.layers[:-4]:
    layer.trainable = False


classifier = K.models.Sequential()
classifier.add(pre_trained)
classifier.add(K.layers.Flatten(input_shape=pre_trained.output_shape[1:]))
classifier.add(K.layers.Dense(256, activation='relu'))
classifier.add(K.layers.Dropout(0.25))
classifier.add(K.layers.Dense(256, activation='relu'))
classifier.add(K.layers.Dropout(0.25))
classifier.add(K.layers.Dense(NUM_CLASSES, activation='softmax'))

classifier.summary()
