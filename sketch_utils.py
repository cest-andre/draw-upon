import ndjson
import random
import base64
import os
from tensorflow import keras
import numpy as np


def fetchSketch(sketchCat):
    print("Loading sketch ndjson.")
    with open('sketch_data/quickdraw_preprocessed_data/full_simplified_' + sketchCat + '.ndjson') as f:
        sketches = ndjson.load(f)

    #response = sketches[0]["drawing"]
    response = []
    strokeHolder = []
    sketchHolder = []
    #   Some googling suggests that only 1k training instances per class is needed.
    # missingSketches = []
    # for i in range(1000):
    #     if not os.path.isdir(f"sketch_data/converted_data/{sketchCat}/{sketchCat}_{i+1}/"):
    #         missingSketches.append(i)

    numSketches = 2000
    #whichSketch = random.randrange(len(sketches))
    #whichSketch = 89154
    #print(f"Which sketch: {whichSketch}")

    print("Reformatting sketches.")
    # for l in range(len(missingSketches)):
    #     i = missingSketches[l]
    for i in range(numSketches):
        #   sketches[which drawing][drawing attribute that contains coords][which stroke][x, y, time][which point]
        for j in range(len(sketches[i]["drawing"])):            #   Each stroke.
            for k in range(len(sketches[i]["drawing"][j][0])):     #   Each coordinate ([0] is x list and [1] is y list.
                strokeHolder.append([sketches[i]["drawing"][j][0][k], sketches[i]["drawing"][j][1][k]])
            sketchHolder.append(strokeHolder)
            strokeHolder = []
        response.append(sketchHolder)
        sketchHolder = []

    return response


def createSketchSequence(data):
    sketchSequence = data["sketchSequence"]
    category = data["sketchCat"]
    sketchIndex = data["sketchIndex"]

    sequencePath = f"sketch_data/converted_data/{category}/{category}_{sketchIndex+1}"
    os.mkdir(sequencePath)

    for i in range(len(sketchSequence)):
        with open(f"{sequencePath}/{i+1}.png", "wb") as f:
            f.write(base64.b64decode(sketchSequence[i]))


def prepareSketchTransferData():
    pass
    # train_list = []
    # test_list = []
    #
    # for img in train_imgs:
    #     train_list.append(img[0])
    #
    # for img in test_imgs:
    #     test_list.append(img[0])
    #
    # np.save('sketch_data/sketchtransfer/sketchtransfer_cnn_train_samples.npy', np.array(train_list, dtype="uint8"))
    # np.save('sketch_data/sketchtransfer/sketchtransfer_cnn_test_samples.npy', np.array(test_list, dtype="uint8"))


    #   For data transformation.
    # tl = []
    # test_labels = np.load('sketch_data/sketchtransfer/quickdraw_y_test.npz.npy', allow_pickle=True)
    # print(test_labels.shape)
    # for i in range(10):
    #     if i == 4:
    #         continue
    #     for j in range(2500):
    #         tl.append(i)
    #
    #
    # print(tl)
    # np.save('sketch_data/sketchtransfer/sketchtransfer_test_labels.npy', np.array(tl, dtype="uint8"))

    #   Transform cifar into sequence and save.
    # train_samples = []
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # # labels = []
    # # for l in y_test:
    # #     labels.append(l[0])
    # # np.save('sketch_data/sketchtransfer/cifar_test_labels', np.array(labels, dtype="uint8"))
    #
    # for x in x_train:
    #     seq_size = 1
    #     holder = []
    #     for i in range(seq_size):
    #         holder.append(x)
    #
    #     train_samples.append(holder)
    #
    #
    # print(np.array(train_samples).shape)
    # np.save('sketch_data/sketchtransfer/sketchtransfer_train_samples_singles', train_samples)

    #   Transform sketchtransfer test data into sequence and save.
    # test_samples = []
    # test_imgs = np.load('sketch_data/sketchtransfer/x_test_compress.npz', allow_pickle=True)["arr_0"]
    # indices = [0, 2500, 15000, 5000, 0, 0, 0, 5000, 2500]
    # c = 0
    # for i in indices:
    #     for j in range(2500):
    #         x = test_imgs[c]
    #         x = x * 255
    #         x = np.transpose(x, [1, 2, 0])
    #         seq_size = 1
    #         holder = []
    #
    #         for k in range(seq_size):
    #             holder.append(x)
    #
    #         test_samples.append(holder)
    #         c += 1
    #     c += i
    #     print(c)
    #
    # print(np.array(test_samples).shape)
    # np.save('sketch_data/sketchtransfer/sketchtransfer_test_samples_singles', np.array(test_samples, dtype="uint8"))


# sketchCat = "tree"
# cats = ""
# for i in range(2000):
#         if not os.path.isdir(f"sketch_data/converted_data/{sketchCat}/{sketchCat}_{i+1}/"):
#             cats = f"{cats} {i},"
#
# print(cats)

# prepareSketchTransferData()