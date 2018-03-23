import os.path as path
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import random


# --Constants--
NUM_CLASSES = 10
LABEL_NAMES = ['Art Nouveau (Modern)', 'Baroque', 'Expressionism',
               'Impressionism', 'Post-Impressionsim', 'Realism',
               'Rococo', 'Romanticism', 'Surrealism', 'Symbolism']
TRAIN_EXAMPLES = 48100  # approx. number of train examples
TEST_EXAMPLES = 14790  # approx. number of validation examples
INPUT_SHAPE = (224, 224, 3)
if path.isdir('/painting-styles'):  # we are on floydhub
    TRAIN_RECORD = '/painting-styles/train.tfrecords'
    TEST_RECORD = '/painting-styles/test.tfrecords'
    OUTPUT_DIR = '/output'
else:
    DATA_DIR = path.expanduser('cnn_input')
    TRAIN_RECORD = path.join(DATA_DIR, 'painting-styles\\train.tfrecords')
    TEST_RECORD = path.join(DATA_DIR, 'painting-styles\\test.tfrecords')
    OUTPUT_DIR = path.expanduser('output')

# --Configurations--
EPOCHS = 20
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
    rand_brightness_chance = random.randint(0, 100)

    if flip_chance <= 20:
        image = tf.image.flip_left_right(image)
    if rand_brightness_chance <= 20:
        image = tf.image.random_brightness(image, 0.4)
    return dict(zip(['resnet50_input'], [K.applications.vgg16.preprocess_input(image)])), label


def batch_generator(path_to_recod, batch_size=BATCH_SIZE, parser_function=parser):
    dataset = tf.data.TFRecordDataset(path_to_recod)
    dataset = (dataset.repeat(EPOCHS + 1)  # so we don't run out of data, even with rounding errors
               .shuffle(1000)
               .map(parser_function)
               .map(preprocessing)
               .batch(batch_size)
               .prefetch(batch_size))
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    while True:
        yield K.backend.get_session().run(next_batch)


tf.logging.set_verbosity(tf.logging.INFO)
pre_trained = K.applications.resnet50.ResNet50(include_top=False,
                                               weights='imagenet',
                                               input_shape=INPUT_SHAPE)

for layer in pre_trained.layers[:-11]:
    layer.trainable = False

model = K.models.Sequential()
model.add(pre_trained)

model.add(K.layers.Flatten(input_shape=pre_trained.output_shape[1:]))

model.add(K.layers.Dense(128, kernel_regularizer=K.regularizers.l2()))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Activation('relu'))

model.add(K.layers.Dense(128, kernel_regularizer=K.regularizers.l2()))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Activation('relu'))

model.add(K.layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer=K.optimizers.Nadam(lr=0.0002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

tensorboardCB = K.callbacks.TensorBoard(log_dir=OUTPUT_DIR,
                                        histogram_freq=0,
                                        write_graph=True)
checkpointCB = K.callbacks.ModelCheckpoint(filepath=OUTPUT_DIR + '/model.h5',
                                           verbose=1,
                                           save_best_only=True,
                                           monitor='val_acc',
                                           mode='max')
earlystoppingCB = K.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=2, verbose=1)

model.fit_generator(generator=batch_generator(TRAIN_RECORD),
                    steps_per_epoch=TRAIN_EXAMPLES / BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=batch_generator(TEST_RECORD),
                    validation_steps=TEST_EXAMPLES / BATCH_SIZE,
                    verbose=2,
                    callbacks=[checkpointCB, tensorboardCB, earlystoppingCB],
                    workers=0)
predictions = model.predict_generator(generator=batch_generator(TEST_RECORD), steps=1, workers=0, verbose=1)
print(predictions)
predictions = np.argmax(predictions, axis=-1)
print(predictions)