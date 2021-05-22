import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #  Suppresses some logs

from io import BytesIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import math
import cv2
from tensorflow.keras import backend as k
from tensorflow.keras import activations
from tensorflow.python.ops import array_ops
import base64

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


class MaskConvLSTM2D(layers.Layer):
    def __init__(self, numFilters, return_seq, kernel_size=3, return_state=False, data_format="channels_last", predict=False, **kwargs):
        super(MaskConvLSTM2D, self).__init__(**kwargs)
        self.convLSTM = layers.ConvLSTM2D(numFilters, kernel_size, padding='same', return_state=return_state, return_sequences=return_seq, data_format=data_format)
        self.predict = predict


    def call(self, inputs, initial_state=None, **kwargs):
        # print("Reminder that I must cast input to float32 when calling layer manually (such as in predict_sketch).")
        # print("Very important to comment the cast when training a network.  Appears to interfere!")

        #   Needed when computing layers manually (such as for confidence dynamics and confusion matrix production).
        if self.predict:
            inputs = tf.cast(inputs, dtype=tf.float32)

        if initial_state is None:
            return self.convLSTM(inputs)#, mask=mask)
        else:
            return self.convLSTM(inputs, initial_state=initial_state)


    def compute_mask(self, inputs, mask=None):
        mask = tf.math.not_equal(inputs, 1000)
        mask = tf.reduce_all(mask, axis=[2, 3, 4])
        return mask


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


def train_lstm_model(train_samples, train_labels, test_samples, test_labels):
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    train_samples = keras.preprocessing.sequence.pad_sequences(train_samples, padding="post", value=1000)
    test_samples = keras.preprocessing.sequence.pad_sequences(test_samples, padding="post", value=1000)

    # train_samples = keras.preprocessing.sequence.pad_sequences(train_samples, padding="post", value=1000*255)
    # test_samples = keras.preprocessing.sequence.pad_sequences(test_samples, padding="post", value=1000*255)

    # train_samples = train_samples.astype("float32") / 255.0
    # test_samples = test_samples.astype("float32") / 255.0
    #
    train_samples = np.reshape(train_samples, [train_samples.shape[0], 16, 32, 32, 1])
    test_samples = np.reshape(test_samples, [test_samples.shape[0], 16, 32, 32, 1])

    img_dim = train_samples.shape[2]
    seq_size = train_samples.shape[1]

    print(train_samples.shape)
    print(train_labels.shape)

    model = keras.Sequential()

    #   Returns output for each timestep so that stacking RNN layers is possible (output fed into next RNN layer).
    #   We only want to pass RNN layer results to the classifier layer at the end of the sequence.  So return_sequences is not set in the second layer.
    model.add(keras.Input(shape=(seq_size, img_dim, img_dim, 1)))

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

    #   Up next.
    model.add(MaskConvLSTM2D(128, return_seq=True, kernel_size=7))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    model.add(MaskConvLSTM2D(256, return_seq=True, kernel_size=5))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    model.add(MaskConvLSTM2D(512, return_seq=False, kernel_size=3))

    #   Maybe add flatten, or global pool all filters.
    #   Could add conv layers here instead.
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(25, activation='softmax'))

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=["accuracy"]
    )

    print(model.summary())

    batch_size = 4

    # #   Try autotune.
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_data =

    # model.load_weights('trained_sketch_rec_models/candidate_2k-14_weights_3ep/')
    epoch = 1
    while epoch < 5:
        model.fit(train_samples, train_labels, batch_size=batch_size, epochs=1)
        model.save_weights(f'trained_sketch_rec_models/candidate_2k-14_weights_{epoch}ep/')
        epoch += 1

    model.evaluate(test_samples, test_labels, batch_size=batch_size)


def train_cnn_model(train_samples, train_labels, test_samples, test_labels):
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

    img_dim = train_samples.shape[2]

    print(train_samples.shape)
    print(train_labels.shape)

    inputs = keras.Input(shape=(32, 32, 1))
    x = layers.Conv2D(128, 9, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 9, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 9, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(256, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(256, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
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

    model.load_weights('trained_sketch_rec_models/cnn_candidate_3_weights_50ep/')
    epoch = 55

    # while epoch < 65:
    #     model.fit(train_samples, train_labels, batch_size=batch_size, epochs=5)
    #     model.save_weights(f'trained_sketch_rec_models/cnn_candidate_3_weights_{epoch}ep/')
    #     epoch += 5

    # model.fit(train_samples, train_labels, batch_size=batch_size, epochs=5)
    # model.save_weights(f'trained_sketch_rec_models/cnn_candidate_4_weights_{epoch}ep/')

    model.evaluate(test_samples, test_labels, batch_size=batch_size)


def matrix_to_png_b64(m):
    img = keras.preprocessing.image.array_to_img(m)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img = buffered.getvalue()
    img = base64.b64encode(img)
    img = img.decode("UTF-8")

    return img


def produce_confusion_matrix(matrix_path):
    #   Obtain test dataset and begin predictions.  Take highest value as prediction, tally up predictions per category (row is ground truth, column is prediction),
    #   then divide all entries by number of instances of each category.  Finally, save the 25x25 tensor to a file.  Convert it into an image and return to predict_sketch
    #   to send it to client.
    test_samples = np.load('sketch_data/test_samples.npy', allow_pickle=True)
    test_labels = np.load('sketch_data/test_labels.npy', allow_pickle=True)
    test_labels = np.asarray(test_labels)

    test_samples = keras.preprocessing.sequence.pad_sequences(test_samples, padding="post", value=1000)
    test_samples = np.reshape(test_samples, [test_samples.shape[0], 16, 32, 32, 1])

    img_dim = test_samples.shape[2]
    seq_size = test_samples.shape[1]

    inputs = keras.Input(shape=(seq_size, img_dim, img_dim, 1))

    #   Model 3.
    # x = MaskConvLSTM2D(128, return_seq=True, kernel_size=5, return_state=False, predict=True)(inputs)
    # x = MaskConvLSTM2D(256, return_seq=False, kernel_size=5, return_state=False, predict=True)(x)

    #   Model 12.
    # x = MaskConvLSTM2D(128, return_seq=True, kernel_size=3, predict=True)(inputs)
    # x = MaskConvLSTM2D(256, return_seq=True, kernel_size=3, predict=True)(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)
    # x = MaskConvLSTM2D(512, return_seq=False, kernel_size=3, predict=True)(x)

    #   Model 14.
    x = MaskConvLSTM2D(128, return_seq=True, kernel_size=7, predict=True)(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)
    x = MaskConvLSTM2D(256, return_seq=True, kernel_size=5, predict=True)(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)
    x = MaskConvLSTM2D(512, return_seq=False, kernel_size=3, predict=True)(x)


    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(25, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights('trained_sketch_rec_models/candidate_2k-14_weights_3ep/')
    print(model.summary())

    conf_matrix = np.zeros([25, 25])

    #   Keep count of number of ground truth instances per category for later division.  I can cheat for now since I know how many there are.
    for i in range(0, test_labels.shape[0], 4):
        if i % 100 == 0:
            print(i)

        input, label = test_samples[i:i+3], test_labels[i:i+3]
        input = tf.stack(input, axis=0)
        layer_out = model.layers[1](input)

        for j in range(2, len(model.layers) - 1):
            layer_out = model.layers[j](layer_out)

        prediction = model.layers[len(model.layers) - 1](layer_out)

        for j in range(prediction.shape[0]):
            conf_matrix[label[j]][tf.argmax(prediction[j]).numpy()] += 1

    conf_matrix = conf_matrix / 400
    conf_min, conf_max = tf.reduce_min(conf_matrix), tf.reduce_max(conf_matrix)

    conf_matrix = tf.cast(((conf_matrix - conf_min) / (conf_max - conf_min))*255, tf.uint8)
    conf_matrix = cv2.resize(conf_matrix.numpy(), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    conf_matrix = tf.constant(conf_matrix, shape=[conf_matrix.shape[0], conf_matrix.shape[1], 1])
    np.save(matrix_path, conf_matrix)

    conf_img = matrix_to_png_b64(conf_matrix)

    return conf_img


#   Functional model setup for state visualization.
def get_models(seq_size, img_dim):
    #   LSTM Model.
    inputs = keras.Input(shape=(seq_size, img_dim, img_dim, 1))

    #   Model 3.
    # x = MaskConvLSTM2D(128, return_seq=True, kernel_size=5, return_state=True, predict=True)(inputs)
    # x = MaskConvLSTM2D(256, return_seq=False, kernel_size=5, return_state=True, predict=True)(x[0])

    #   Model 12.
    # x = MaskConvLSTM2D(128, return_seq=True, kernel_size=3, return_state=True, predict=True)(inputs)
    # x = MaskConvLSTM2D(256, return_seq=True, kernel_size=3, return_state=True, predict=True)(x[0])
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x[0])
    # x = MaskConvLSTM2D(512, return_seq=False, kernel_size=3, return_state=True, predict=True)(tf.stack([x[0]], axis=0))

    #   Model 14.
    x = MaskConvLSTM2D(128, return_seq=True, kernel_size=7, return_state=True, predict=True)(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x[0])
    x = MaskConvLSTM2D(256, return_seq=True, kernel_size=5, return_state=True, predict=True)(tf.stack([x[0]], axis=0))
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x[0])
    x = MaskConvLSTM2D(512, return_seq=False, kernel_size=3, return_state=True, predict=True)(tf.stack([x[0]], axis=0))

    x = layers.GlobalAveragePooling2D()(x[0])
    outputs = layers.Dense(25, activation='softmax')(x)

    lstm_model = keras.Model(inputs=inputs, outputs=outputs)
    # model.load_weights('trained_sketch_rec_models/candidate_2k-3_weights_3ep/')
    # model.load_weights('trained_sketch_rec_models/candidate_2k-12_weights_5ep/')
    lstm_model.load_weights('trained_sketch_rec_models/candidate_2k-14_weights_3ep/')
    print(lstm_model.summary())

    #   CNN Model.
    inputs = keras.Input(shape=(img_dim, img_dim, 1))

    #   Model 3.
    x = layers.Conv2D(128, 9, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 9, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 9, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(256, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(256, 7, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(512, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
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
    outputs = layers.Dense(25, activation='softmax')(x)

    cnn_model = keras.Model(inputs=inputs, outputs=outputs)
    cnn_model.load_weights('trained_sketch_rec_models/cnn_candidate_3_weights_50ep/')

    print(cnn_model.summary())

    return lstm_model, cnn_model


def predict_sketch(sketch_sequence):
    numFrames = len(sketch_sequence)
    seqLength = 16
    frameNum = 0
    frameInc = math.ceil(numFrames / seqLength)
    sketch_imgs = []
    returned_imgs = []

    print(f"Sequence length: {numFrames}")
    while frameNum < numFrames:
        img = base64.b64decode(sketch_sequence[frameNum])
        img = tf.image.decode_png(img, channels=4).numpy()
        #   Black sketch on white background.
        #img = (255 - img[:, :, 3])
        #   White sketch on black background.
        img = img[:, :, 3]
        #   Try other resizers.
        img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, [32, 32, 1])
        #tfimg = tf.constant(img)
        #img = img.astype("float32")
        sketch_imgs.append(img)
        returned_imgs.append(matrix_to_png_b64(img))

        if frameNum == numFrames - 1:
            break
        elif frameNum + frameInc > numFrames - 1:
            frameNum = numFrames - 1
        else:
            frameNum += frameInc

    print(len(sketch_imgs))
    wrapper = [np.stack(sketch_imgs, axis=0)]

    sample = keras.preprocessing.sequence.pad_sequences(wrapper, maxlen=16, padding="post", value=1000)

    #   Make sure you know the architecture of the layer before executing this.  Need to change when number of layers change.
    lstm_model, cnn_model = get_models(sample.shape[1], sample.shape[2])

    inputs = lstm_model.layers[0](sample)
    returned_states = []
    predictions = []

    #   For each convLSTM layer in model, hidden states from previous timestep will be stored in hidden_states.
    hidden_states = []
    for i in range(len(lstm_model.layers)):
        if "mask_conv_lst_m2d" in lstm_model.layers[i].name:
            hidden_states.append(None)

    #   Double stack inputs before feeding to convLSTM layers because they expect dim = 5 (batch size, sequence size, img width, img height, channels).
    for i in range(len(sketch_imgs)):
        #   Calculate feedforward CNN prediction first.
        cnn_in = tf.stack([sketch_imgs[i]], axis=0)
        cnn_in = tf.cast(cnn_in, dtype=tf.float32)

        cnn_prediction = cnn_model.layers[0](cnn_in)
        for j in range(1, len(cnn_model.layers)):
            cnn_prediction = cnn_model.layers[j](cnn_prediction)

        #   Time for LSTM.
        input = tf.stack([inputs[0][i]], axis=0)
        #   Calculate the fist filter's input gate for the first layer.
        kernel_i, kernel_f, kernel_c, kernel_o = array_ops.split(lstm_model.layers[1].weights[0], 4, axis=3)
        r_kernel_i, r_kernel_f, r_kernel_c, r_kernel_o  = array_ops.split(lstm_model.layers[1].weights[1], 4, axis=3)
        bias_i, bias_f, bias_c, bias_o = array_ops.split(lstm_model.layers[1].weights[2], 4)

        i_gate = None
        x_i = k.conv2d(tf.cast(input, tf.float32), kernel_i, padding="same")
        x_i = k.bias_add(x_i, bias_i)
        if hidden_states[0] is None:
            i_gate = activations.sigmoid(x_i)
        else:
            h_i = k.conv2d(hidden_states[0][1], r_kernel_i, padding='same')
            i_gate = activations.sigmoid(x_i + h_i)
        i_gate = tf.transpose(i_gate, [3, 1, 2, 0])[0]
        i_min, i_max = tf.reduce_min(i_gate), tf.reduce_max(i_gate)

        gate_state = tf.cast(((i_gate - i_min) / (i_max - i_min))*255, tf.uint8)
        gate_img = matrix_to_png_b64(gate_state.numpy())

        pooled_states = None
        hidden_count = 0

        for j in range(1, len(lstm_model.layers) - 2):
            wrapped_in = None
            if j == 1:
                wrapped_in = input
                wrapped_in = tf.stack([wrapped_in], axis=0)
            elif pooled_states is not None:
                wrapped_in = pooled_states
                pooled_states = None
            else:
                wrapped_in = hidden_states[hidden_count-1][0]

            if "mask_conv_lst_m2d" in lstm_model.layers[j].name:
                if hidden_states[hidden_count] is None:
                    hidden_states[hidden_count] = lstm_model.layers[j](wrapped_in)
                else:
                    hidden_states[hidden_count] = lstm_model.layers[j](wrapped_in, [hidden_states[hidden_count][1], hidden_states[hidden_count][2]])

                hidden_count += 1
            else:
                pooled_states = lstm_model.layers[j](wrapped_in)

        last_state = None

        if pooled_states is None:
            last_state = hidden_states[len(hidden_states) - 1]
        else:
            last_state = pooled_states

        out_state = lstm_model.layers[len(lstm_model.layers) - 2](tf.stack([last_state[0][0]], axis=0))
        prediction = lstm_model.layers[len(lstm_model.layers) - 1](out_state)

        #   Transform first cell and completed filter from second layer into an image.
        filter_state = tf.transpose(hidden_states[0][1], [0, 3, 1, 2])
        cell_state = tf.transpose(hidden_states[0][2], [0, 3, 1, 2])
        #   Second dim of h_state decides which cell to choose.  Each output filter has a cell.
        filter_state = tf.reshape(filter_state[0][1], shape=[32, 32, 1])
        cell_state = tf.reshape(cell_state[0][1], shape=[32, 32, 1])

        f_min, f_max = tf.reduce_min(filter_state), tf.reduce_max(filter_state)
        c_min, c_max = tf.reduce_min(cell_state), tf.reduce_max(cell_state)

        filter_state = tf.cast(((filter_state - f_min) / (f_max - f_min))*255, tf.uint8)
        cell_state = tf.cast(((cell_state - c_min) / (c_max - c_min))*255, tf.uint8)

        filter_img = matrix_to_png_b64(filter_state.numpy())
        cell_img = matrix_to_png_b64(cell_state.numpy())

        returned_states.append([gate_img, cell_img, filter_img])
        predictions.append({"lstm": prediction.numpy().tolist(), "cnn": cnn_prediction.numpy().tolist()})

    #   Construct confusion images.
    conf_imgs = []

    model_3_conf = np.load('confusion/model_2k-3_3ep.npy', allow_pickle=True)
    model_12_conf = np.load('confusion/model_2k-12_5ep.npy', allow_pickle=True)
    model_14_conf = np.load('confusion/model_2k-14_3ep.npy', allow_pickle=True)

    conf_imgs.append({"caption": "Model 3 Confusion Matrix", "img": matrix_to_png_b64(model_3_conf)})
    conf_imgs.append({"caption": "Model 12 Confusion Matrix", "img": matrix_to_png_b64(model_12_conf)})
    conf_imgs.append({"caption": "Model 14 Confusion Matrix", "img": matrix_to_png_b64(model_14_conf)})

    #   Compute differences.
    # diffs = model_12_conf - model_3_conf
    # diff_min, diff_max = tf.reduce_min(diffs), tf.reduce_max(diffs)
    #
    # conf_matrix = tf.cast(((diffs - diff_min) / (diff_max - diff_min))*255, tf.uint8)
    # conf_matrix = cv2.resize(conf_matrix.numpy(), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    # conf_matrix = tf.constant(conf_matrix, shape=[conf_matrix.shape[0], conf_matrix.shape[1], 1])
    #
    # conf_imgs.append({"caption": "12 minus 3 Confusion Matrix", "img": matrix_to_png_b64(conf_matrix)})
    #
    # diffs = model_14_conf - model_12_conf
    # diff_min, diff_max = tf.reduce_min(diffs), tf.reduce_max(diffs)
    #
    # conf_matrix = tf.cast(((diffs - diff_min) / (diff_max - diff_min))*255, tf.uint8)
    # conf_matrix = cv2.resize(conf_matrix.numpy(), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    # conf_matrix = tf.constant(conf_matrix, shape=[conf_matrix.shape[0], conf_matrix.shape[1], 1])
    #
    # conf_imgs.append({"caption": "14 minus 12 Confusion Matrix", "img": matrix_to_png_b64(model_14_conf - model_12_conf)})
    # conf_imgs.append({"caption": "14 minus 3 Confusion Matrix", "img": matrix_to_png_b64(model_14_conf - model_3_conf)})

    #   Send off hidden states such as input and output gates for attention visualization as well.
    return {"catList": cat_list, "predictions": predictions, "inputs": returned_imgs, "states": returned_states, "confusions": conf_imgs}


#   Current category selection is preferred because they contain much closer to 1k sub-50 length sequences.  Need to recreate dataset using 2 segments
#   per frame to shorten these sequences and have more instances.
# train_samples, train_labels, test_samples, test_labels = load_cnn_data("sketch_data/converted_data/", cat_list)
# train_cnn_model(train_samples, train_labels, test_samples, test_labels)

# produce_confusion_matrix('confusion/model_2k-14_3ep.npy')
