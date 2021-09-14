import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #  Suppresses some logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
import tensorflow_addons as tfa
import numpy as np
import math
import cv2
from sketch_model import MaskConvLSTM2D, ConvLSTMModel, ConvLSTM2DCellNormed
from PIL import Image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#   Disables GPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#   Sets memory growth to be ad hoc so that I can observe GPU memory usage (if this is not done, Tensorflow will automatically allocate all GPU memory even if unneeded).
#   It is useful to observe GPU memory demands to guide my network size and batch size.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# cat_list = [
#     "ant", "banana", "bicycle", "brain", "car",
#     "cat", "coffee cup", "computer", "guitar", "helicopter",
#     "house", "key", "light bulb", "map", "mosquito",
#     "mug", "mushroom", "onion", "pizza", "river",
#     "saxophone", "shoe", "skateboard", "tree", "wine glass"
# ]

#   Removing categories that are too geometrically similar such as mug/coffee cup, mushroom/tree, and mosquito/ant.
cat_list = [
    "ant", "banana", "bicycle", "brain", "car",
    "cat", "coffee cup", "computer", "eye", "guitar",
    "hammer", "helicopter", "hourglass", "house", "key",
    "light bulb", "map", "onion", "pizza", "river",
    "saxophone", "shoe", "skateboard", "tree", "wine glass"
]


def load_lstm_sketchtransfer_data():
    #   Load data.
    train_imgs = np.load('sketch_data/sketchtransfer/sketchtransfer_train_samples.npy', allow_pickle=True)
    #   May need to invert sketches to be white on black background.
    test_imgs = np.load('sketch_data/sketchtransfer/sketchtransfer_test_samples.npy', allow_pickle=True)
    train_labels = np.load('sketch_data/sketchtransfer/sketchtransfer_train_labels.npy', allow_pickle=True)
    test_labels = np.load('sketch_data/sketchtransfer/sketchtransfer_test_labels.npy', allow_pickle=True)

    return train_imgs, train_labels, test_imgs, test_labels


def load_cnn_sketchtransfer_data():
    #   Load data.
    train_imgs = np.load('sketch_data/sketchtransfer/sketchtransfer_cnn_train_samples.npy', allow_pickle=True)
    #   May need to invert sketches to be white on black background.
    test_imgs = np.load('sketch_data/sketchtransfer/sketchtransfer_cnn_test_samples.npy', allow_pickle=True)
    train_labels = np.load('sketch_data/sketchtransfer/sketchtransfer_train_labels.npy', allow_pickle=True)
    test_labels = np.load('sketch_data/sketchtransfer/sketchtransfer_test_labels.npy', allow_pickle=True)

    return train_imgs, train_labels, test_imgs, test_labels


def load_lstm_data(prefix, categories):
    if (os.path.isfile('sketch_data/train_samples.npy') and os.path.isfile('sketch_data/train_labels.npy') and
        os.path.isfile('sketch_data/test_samples.npy') and os.path.isfile('sketch_data/test_labels.npy')):
        train_samples = np.load('sketch_data/train_samples.npy', allow_pickle=True)
        train_labels = np.load('sketch_data/train_labels.npy', allow_pickle=True)
        test_samples = np.load('sketch_data/test_samples.npy', allow_pickle=True)
        test_labels = np.load('sketch_data/test_labels.npy', allow_pickle=True)

        return train_samples, train_labels, test_samples, test_labels

    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    total_samples = 2000
    num_training_per_cat = 1600

    for i in range(len(categories)):
        category = categories[i]
        label = i
        catCount = 0

        for j in range(total_samples):
            frameNum = 1
            #   Drop to 15 so that max is 16 due to final frame getting appended in addition to seqLength frames for some samples.
            seqLength = 15
            seqDir = prefix + f"{category}/{category}_{j+1}/"
            seqData = []
            numFrames = len(os.listdir(seqDir))
            frameInc = math.ceil(numFrames / seqLength)

            # if os.path.isfile(seqDir + "50.png"):
            #     continue
            while os.path.isfile(seqDir + f"{frameNum}.png"):
                #   Images are originally in RGBA (a is "alpha" and determines opacity of pixel.  The higher to opacity, the less transparent the pixel color).
                #   For some reason, the images captured by my paper path to raster are all black pixels (0, 0, 0) with the A channel controlling the grayscale
                #   (0 opacity means fully transparent which is white, A = 255 is fully black).  This numbering scheme is the inverse of grayscale where
                #   0 is black and white is 1 I think.

                #   Perhaps make np array so that more GRAM is available.
                img = tf.io.read_file(seqDir + f"{frameNum}.png")
                img = tf.image.decode_png(img, channels=4).numpy()
                #   Black sketch on white background.
                #img = (255 - img[:, :, 3])
                #   White sketch on black background.
                img = img[:, :, 3]
                #   Try other resizers.
                img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
                img = np.reshape(img, [32, 32, 1])
                seqData.append(img)
                if frameNum == numFrames:
                    break
                elif frameNum + frameInc > numFrames:
                    frameNum = numFrames
                else:
                    frameNum += frameInc
                #   Detect if I passed up the last frame.  Maybe try to obtain count of folders in and set frameNum to be the last number if it isn't currently or is greater than (before increment).

            # while len(seqData) < 400:
            #     seqData.append(tf.zeros(shape=[256, 256, 1], dtype=tf.float32))
            #samples.append(tf.stack(seqData, axis=0))
            if j < num_training_per_cat:
                train_samples.append(np.stack(seqData, axis=0))
                train_labels.append(label)
            else:
                test_samples.append(np.stack(seqData, axis=0))
                test_labels.append(label)

            catCount += 1
            if catCount % 100 == 0:
                print(category, catCount)

            # if catCount >= 1000:
            #     break

    np.save('sketch_data/train_samples.npy', train_samples)
    np.save('sketch_data/train_labels.npy', train_labels)
    np.save('sketch_data/test_samples.npy', test_samples)
    np.save('sketch_data/test_labels.npy', test_labels)

    return train_samples, train_labels, test_samples, test_labels


def load_cnn_data(prefix, categories):
    if (os.path.isfile('sketch_data/cnn_samples/train_samples.npy') and os.path.isfile('sketch_data/cnn_samples/train_labels.npy') and
        os.path.isfile('sketch_data/cnn_samples/test_samples.npy') and os.path.isfile('sketch_data/cnn_samples/test_labels.npy')):
        train_samples = np.load('sketch_data/cnn_samples/train_samples.npy', allow_pickle=True)
        train_labels = np.load('sketch_data/cnn_samples/train_labels.npy', allow_pickle=True)
        test_samples = np.load('sketch_data/cnn_samples/test_samples.npy', allow_pickle=True)
        test_labels = np.load('sketch_data/cnn_samples/test_labels.npy', allow_pickle=True)

        return train_samples, train_labels, test_samples, test_labels

    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    total_samples = 2000
    num_training_per_cat = 1600

    for i in range(len(categories)):
        category = categories[i]
        label = i
        catCount = 0

        for j in range(total_samples):
            #   Drop to 15 so that max is 16 due to final frame getting appended in addition to seqLength frames for some samples.
            seqDir = prefix + f"{category}/{category}_{j+1}/"
            numFrames = len(os.listdir(seqDir))

            img = tf.io.read_file(seqDir + f"{numFrames}.png")
            img = tf.image.decode_png(img, channels=4).numpy()
            #   Black sketch on white background.
            #img = (255 - img[:, :, 3])
            #   White sketch on black background.
            img = img[:, :, 3]
            #   Try other resizers.
            img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
            img = np.reshape(img, [32, 32, 1])

            if j < num_training_per_cat:
                train_samples.append(img)
                train_labels.append(label)
            else:
                test_samples.append(img)
                test_labels.append(label)

            catCount += 1
            if catCount % 100 == 0:
                print(category, catCount)

            # if catCount >= 1000:
            #     break

    np.save('sketch_data/cnn_samples/train_samples.npy', train_samples)
    np.save('sketch_data/cnn_samples/train_labels.npy', train_labels)
    np.save('sketch_data/cnn_samples/test_samples.npy', test_samples)
    np.save('sketch_data/cnn_samples/test_labels.npy', test_labels)

    return train_samples, train_labels, test_samples, test_labels


def train_lstm_model(train_samples, train_labels, test_samples, test_labels, transfer=False):
    if not transfer:
        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)

        train_samples = keras.preprocessing.sequence.pad_sequences(train_samples, padding="post", value=1000)
        test_samples = keras.preprocessing.sequence.pad_sequences(test_samples, padding="post", value=1000)

        train_samples = np.reshape(train_samples, [train_samples.shape[0], 16, 32, 32, 1])
        test_samples = np.reshape(test_samples, [test_samples.shape[0], 16, 32, 32, 1])
    # else:
    #     train_samples = (train_samples.astype("float32")) / 255.0
    #     cv2.imshow('beep', train_samples[0][0])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return


    # train_samples = keras.preprocessing.sequence.pad_sequences(train_samples, padding="post", value=1000*255)
    # test_samples = keras.preprocessing.sequence.pad_sequences(test_samples, padding="post", value=1000*255)

    train_samples = train_samples.astype("float32") / 255.0
    test_samples = test_samples.astype("float32") / 255.0

    seq_size = train_samples.shape[1]
    img_dim = train_samples.shape[2]
    channels = train_samples.shape[4]

    print(train_samples.shape)
    print(train_labels.shape)

    # model = keras.Sequential()

    #   Returns output for each timestep so that stacking RNN layers is possible (output fed into next RNN layer).
    #   We only want to pass RNN layer results to the classifier layer at the end of the sequence.  So return_sequences is not set in the second layer.
    # model.add(keras.Input(shape=(seq_size, img_dim, img_dim, channels)))

    #   Model candidate 1 that I will save.
    # model.add(MaskConvLSTM2D(128, return_seq=True, kernel_size=5))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=5))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # # model.add(layers.AveragePooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(256, return_seq=False, kernel_size=3))

    #   Model candidate 2:  still works, but just slightly too big.  Batch size was super low I guess when it worked.  Needs batch size 2 to run.
    # model.add(MaskConvLSTM2D(256, return_seq=True))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(256, return_seq=False))

    #   Model candidate 3: 2 lite
    # model.add(MaskConvLSTM2D(128, return_seq=True, kernel_size=5))
    # model.add(MaskConvLSTM2D(256, return_seq=False, kernel_size=5))

    #   Model 12.
    # model.add(MaskConvLSTM2D(128, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(512, return_seq=False, kernel_size=3))

    #   Model r13.
    # model.add(MaskConvLSTM2D(64, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(128, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(128, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(512, return_seq=False, kernel_size=3))

    #   Model 18.
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=False, kernel_size=3))

    #   Model 19.  This seems to learn.
    # model.add(MaskConvLSTM2D(512, return_seq=True, kernel_size=3))
    # model.add(layers.LayerNormalization())
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(MaskConvLSTM2D(512, return_seq=True, kernel_size=3))
    # model.add(layers.LayerNormalization())
    # model.add(layers.MaxPooling3D(pool_size=(1, 4, 4), padding='same'))

    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=3))
    # model.add(MaskConvLSTM2D(256, return_seq=False, kernel_size=3))

    #   Testing my custom cell.
    # model.add(ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=False))

    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(ConvRNN2D(ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last"), return_sequences=True))
    # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(ConvRNN2D(ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last"), return_sequences=False))

    # model.add(ConvRNN2D(ConvLSTM2DCellNormed(128, 3, padding='same', data_format="channels_last"), return_sequences=True))
    # # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(ConvRNN2D(ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last"), return_sequences=True))
    # # model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    # model.add(ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=False))

    # #   Maybe add flatten, or global pool all filters.
    # #   Could add conv layers here instead.
    # model.add(layers.GlobalAveragePooling2D())
    # if transfer:
    #     model.add(layers.Dense(10, activation='softmax'))
    # else:
    #     model.add(layers.Dense(25, activation='softmax'))

    model = ConvLSTMModel()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=["accuracy"]
    )

    # print(model.summary())

    batch_size = 4

    # #   Try autotune.
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_data =

    # model.load_weights('trained_sketch_rec_models/SketchTransferCandidates/candidate_25a_weights_5ep/')
    epoch = 1
    while epoch < 6:
        model.fit(train_samples, train_labels, batch_size=batch_size, epochs=1)
        model.save_weights(f'trained_sketch_rec_models/SketchTransferCandidates/candidate_27_weights_{epoch}ep/')
        epoch += 1

        model.evaluate(test_samples, test_labels, batch_size=batch_size)


def train_cnn_model(train_samples, train_labels, test_samples, test_labels, transfer=False):
    if not transfer:
        train_samples = np.stack(train_samples, axis=0)
        test_samples = np.stack(test_samples, axis=0)
        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)

        # train_samples = keras.preprocessing.sequence.pad_sequences(train_samples, padding="post", value=1000*255)
        # test_samples = keras.preprocessing.sequence.pad_sequences(test_samples, padding="post", value=1000*255)

        # train_samples = train_samples.astype("float32") / 255.0
        # test_samples = test_samples.astype("float32") / 255.0
        #
        train_samples = np.reshape(train_samples, [train_samples.shape[0], 32, 32, 1])
        test_samples = np.reshape(test_samples, [test_samples.shape[0], 32, 32, 1])
    # else:
    #     train_samples = train_samples / 255
        # print(train_samples[0])
        # cv2.imshow('beep', train_samples[0])
        # print(train_labels[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return

    img_dim = train_samples.shape[2]
    channels = train_samples.shape[3]

    print(train_samples.shape)
    print(train_labels.shape)

    inputs = keras.Input(shape=(img_dim, img_dim, channels))
    x = layers.Conv2D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(256, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(256, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(1024, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(1024, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(1024, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.GlobalAveragePooling2D()(x)

    if transfer:
        outputs = layers.Dense(10, activation='softmax')(x)
    else:
        outputs = layers.Dense(25, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=["accuracy"]
    )

    print(model.summary())

    batch_size = 8

    # #   Try autotune.
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_data =

    model.load_weights('trained_sketch_rec_models/SketchTransferCandidates/cnn_candidate_3_weights_10ep/')
    epoch = 15

    while epoch < 45:
        model.fit(train_samples, train_labels, batch_size=batch_size, epochs=5)
        model.save_weights(f'trained_sketch_rec_models/SketchTransferCandidates/cnn_candidate_3_weights_{epoch}ep/')
        epoch += 5

    # model.fit(train_samples, train_labels, batch_size=batch_size, epochs=5)
    # model.save_weights(f'trained_sketch_rec_models/cnn_candidate_4_weights_{epoch}ep/')

        model.evaluate(test_samples, test_labels, batch_size=batch_size)


#   Current category selection is preferred because they contain much closer to 1k sub-50 length sequences.  Need to recreate dataset using 2 segments
#   per frame to shorten these sequences and have more instances.
# train_samples, train_labels, test_samples, test_labels = load_lstm_data("sketch_data/converted_data/", cat_list)
train_samples, train_labels, test_samples, test_labels = load_lstm_sketchtransfer_data()
train_lstm_model(train_samples, train_labels, test_samples, test_labels, transfer=True)

# train_samples, train_labels, test_samples, test_labels = load_cnn_data("sketch_data/converted_data/", cat_list)
# train_samples, train_labels, test_samples, test_labels = load_cnn_sketchtransfer_data()
# train_cnn_model(train_samples, train_labels, test_samples, test_labels, transfer=True)

# produce_confusion_matrix('confusion/model_2k-14_3ep.npy')