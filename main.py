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
if path.isdir('/painting-styles'):  # we are on floydhub
    TRAIN_RECORD = '/painting-styles/train.tfrecords'
    TEST_RECORD = '/painting-styles/test.tfrecords'
    OUTPUT_DIR = '/output'
else:
    DATA_DIR = path.expanduser('cnn_input')
    TRAIN_RECORD = path.join(DATA_DIR, 'painting-styles\\train.tfrecords')
    TRAIN_RECORD = path.join(DATA_DIR, 'painting-styles\\test.tfrecords')
    OUTPUT_DIR = path.expanduser('output')

# --Configurations--
EPOCHS = 10
BATCH_SIZE = 128


# --Utility functions--
# convert the tfrecord's elements back to usable format
def parser(example_proto):
    features = {'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image': tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(serialized=example_proto, features=features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)
    return image, tf.one_hot(parsed_features['label'], NUM_CLASSES)


# augmentation and preprocessing
def preprocessing(image, label):
    flip_chance = random.randint(0, 100)
  #  rand_brightness_chance = random.randint(0, 100)

    if flip_chance <= 20:
        image = tf.image.flip_left_right(image)
   # if rand_brightness_chance <= 20:
     #   image = tf.image.random_brightness(image, 0.4)
    return dict(zip(['vgg16_input'], [K.applications.vgg16.preprocess_input(image)])), label


def input_pipeline(path_to_recod, batch_size=BATCH_SIZE, parser_function=parser):
    dataset = tf.data.TFRecordDataset(path_to_recod)
    dataset = (dataset.repeat(EPOCHS)
               .shuffle(1000)
               .map(parser_function)
               .map(preprocessing)
               .batch(batch_size)
               .prefetch(batch_size))
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


pre_trained = K.applications.vgg16.VGG16(include_top=False,
                                         weights='imagenet',
                                         input_shape=INPUT_SHAPE)

# Freeze the layers except the last 4 layers
for layer in pre_trained.layers:
    layer.trainable = False

model = K.models.Sequential()
model.add(pre_trained)
model.add(K.layers.Flatten(input_shape=pre_trained.output_shape[1:]))
model.add(K.layers.Dense(1024, activation='relu'))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.6))
model.add(K.layers.Dense(1024, activation='relu'))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.6))
model.add(K.layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer=K.optimizers.Nadam(lr=0.0001, schedule_decay=0.006),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=OUTPUT_DIR)

train_input = lambda: input_pipeline(TRAIN_RECORD)
train_spec = tf.estimator.TrainSpec(input_fn=train_input)
test_input = lambda: input_pipeline(TEST_RECORD)
eval_spec = tf.estimator.EvalSpec(input_fn=test_input)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)