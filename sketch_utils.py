import ndjson
import random
import base64
import os
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt, image
import cv2 as cv
import tensorflow as tf
from PIL import Image

#   Disables GPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#   Data augmentation code.
#   Copied from: https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/tf2/data_util.py
def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.
  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.cast(kernel_size / 2, dtype=tf.int32)
  kernel_size = radius * 2 + 1
  x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
  blur_filter = tf.exp(-tf.pow(x, 2.0) /
                       (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = len(image.shape) == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def color_jitter(image, strength):
  """Distorts the color of the image.
  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.
  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength

  return color_jitter_nonrand(
    image, brightness, contrast, saturation, hue)


def color_jitter_nonrand(image,
                         brightness=0,
                         contrast=0,
                         saturation=0,
                         hue=0):
  """Distorts the color of the image (jittering order is fixed).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.
  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = tf.image.random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def augment_image(x):
    gaussian_noise = keras.layers.GaussianNoise(1)

    #   Preprocess, color distort, blur, then noise.

    # x = tf.cast(x, tf.float32)
    # x = keras.applications.vgg16.preprocess_input(x)

    # x = gaussian_blur(x, 3, 1)

    p = np.random.uniform()
    if p < 0.75:
        p = np.random.uniform()
        if p < 0.8:
            x = color_jitter(x, 1)
        else:
            x = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

        p = np.random.uniform(0.1, 2.0)
        x = gaussian_blur(x, 3, p)

        x = gaussian_noise(x)

    # else:
    #     x = x / 255

    # x = keras.applications.vgg16.preprocess_input(x*255)

    return x


def augment_dataset(x):
    gaussian_noise = keras.layers.GaussianNoise(1)
    inp_list = []

    #   Preprocess, color distort, blur, then noise.
    for i in range(x.shape[0]):
        # if i % 10000 == 0:
        #     print(i)
        # print(i)

        # inp = tf.cast(x[i], tf.float32)
        # inp = tf.identity(x[i])
        inp = x[i]

        # inp = keras.applications.vgg16.preprocess_input(inp)

        p = np.random.uniform()
        if p < 0.5:
            p = np.random.uniform()
            if p < 0.8:
                inp = color_jitter(inp, 1)
            else:
                inp = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(inp))

            p = np.random.uniform(0.1, 2.0)
            inp = gaussian_blur(inp, 3, p)
            inp = gaussian_noise(inp)

        inp = keras.applications.vgg16.preprocess_input(inp)
        inp_list.append(inp)

        #   Maybe run this line instead for testing so that test data is not augmented.
        # inp_list.append(x[i])

    return tf.stack(inp_list, axis=0)


def canny_dataset(x):
    for i in range(x.shape[0]):
        if i % 10000 == 0:
            print(i)

        img = cv.Canny(x[i].astype("uint8"), 100, 200)
        img = 255 - img
        img = tf.expand_dims(img, axis=2)
        img = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img))

        x[i] = keras.applications.vgg16.preprocess_input(img)

        # cv.imshow('beep', x[i])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return x


def get_kernel_variance(kernels):
    kernels = tf.transpose(kernels, perm=[2, 3, 0, 1])
    mean, var = tf.nn.moments(kernels, axes=[2, 3])

    # #   Obtain indices from vars where vars[i] >= threshold.  If so, run these kernels through the blur.
    # for i in range(var.shape[0]):
    #     for j in range(var.shape[1]):
    #         v = var[i][j]
    #         print(v)

    return var


def visualize_kernels(kernels):
    kernels = tf.transpose(kernels, perm=[3, 0, 1, 2]).numpy()

    for i in range(kernels.shape[0]):
        k = kernels[i]
        mean, var = tf.nn.moments(tf.convert_to_tensor(k), axes=[0, 1])
        print(var)

        k_min, k_max = k.min(), k.max()
        k = (k - k_min) / (k_max - k_min)
        k = cv.resize(k, (512,512))

        # kernel = cv.Laplacian(kernels[i][:, :, 0], cv.CV_32F, ksize=3, scale=0.25)
        # kernel = -1 * kernel
        # mean, var = tf.nn.moments(tf.convert_to_tensor(kernel), axes=[0, 1])
        # print(var)

        # k_min, k_max = kernel.min(), kernel.max()
        # kernel = (kernel - k_min) / (k_max - k_min)
        # kernel = cv.resize(kernel, (512,512))

        # f = np.fft.fft2(kernels[i])
        # fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20*np.log(np.abs(fshift))
        # magnitude_spectrum = cv.resize(magnitude_spectrum, (512,512))

        # for j in range(0, k.shape[2], 8):
        #     cv.imshow(str(i) + " channel: " + str(j), k[:,:,j])
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()

        cv.imshow(str(i), k)
        # cv.imshow("Freq Spectrum", magnitude_spectrum)
        # cv.imshow("Higher Freq", kernel)

        cv.waitKey(0)
        cv.destroyAllWindows()


def smooth_V1(model):
    kernels = [
        model.V1.conv_inp.get_weights()[0],
        model.V1.w_gate_exc.read_value(),
        model.V1.w_gate_inh.read_value(),
    ]

    for i in range(len(kernels)):
        kernels[i] = tf.transpose(kernels[i], perm=[3, 0, 1, 2])
        print(kernels[i].shape)


    new_kernels = [
        np.zeros(kernels[0].shape),
        np.zeros(kernels[1].shape, dtype="float32"),
        np.zeros(kernels[2].shape, dtype="float32")
    ]

    for i in range(len(kernels)):
        kernel = kernels[i]
        for j in range(kernel.shape[0]):
            new_kernels[i][j] = gaussian_blur(kernel[j], 11, 0.1)

        new_kernels[i] = tf.transpose(new_kernels[i], perm=[1, 2, 3, 0])

    model.V1.conv_inp.set_weights([tf.convert_to_tensor(new_kernels[0])])
    model.V1.w_gate_exc.assign(tf.convert_to_tensor(new_kernels[1]))
    model.V1.w_gate_inh.assign(tf.convert_to_tensor(new_kernels[2]))


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

    #   Remove deer here (class #4).
    # imgs = np.load('sketch_data/sketchtransfer/sketchtransfer_train_samples_singles.npy', allow_pickle=True)
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    #
    # # labels = np.load('sketch_data/sketchtransfer/sketchtransfer_train_labels.npy')
    # # labels = y_test
    # delete_list = []
    #
    # for i in range(len(y_test)):
    #     if y_test[i] == 4:
    #         delete_list.append(i)
    #
    # print(len(delete_list))
    # x_test = np.delete(x_test, delete_list, 0)
    # y_test = np.delete(y_test, delete_list)
    #
    #
    #
    # for i in range(len(y_test)):
    #     if y_test[i] > 4:
    #         y_test[i] = y_test[i] - 1
    #
    # # print(labels[10000])
    # # plt.imshow(imgs[10000][0])
    # # plt.show()
    #
    # np.save('S:/Documents/Computational Visual Abstraction Project/datasets/sketch_data/sketchtransfer/cifar_cnn_test_samples', np.array(x_test, dtype="uint8"))
    # np.save('S:/Documents/Computational Visual Abstraction Project/datasets/sketch_data/sketchtransfer/cifar_cnn_test_labels', np.array(y_test, dtype="uint8"))


#   I could add a 2 option for much more abstract instances of categories.  These sketches can be very minimal.
def sketch_selector():
    # img = image.imread(r"S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\train\n02123045\n02123045_9.JPEG")
    # plot = plt.imshow(img)

    # img = Image.open(r"S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\train\n02123045\n02123045_9.JPEG")
    # img.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\kittycat.jpg', "JPEG")

    # idx = len(os.listdir(f'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\converted_data\\ant\\ant_{1}'))
    # img = Image.open(f'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\converted_data\\ant\\ant_{1}\\{idx}.png')
    # img.show()

    cat = "shoe"
    cat_count = 0
    for i in range(2000):
        idx = len(os.listdir(f'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\converted_data\\{cat}\\{cat}_{i+1}'))
        img = Image.open(f'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\converted_data\\{cat}\\{cat}_{i+1}\\{idx}.png')
        img.show()
        x = input()
        if x == '1':
            img.save(f'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\\test\\{cat}\\{cat}_{i+1}.png', "PNG")
            if cat_count == 199:
                return
            else:
                cat_count += 1
                print(cat_count)


#   Use below lines to convert.  Build up a list of these arrays along side labels (ascending alphabetically with category name).  Then save to npy file for later retrieval.
# img = tf.io.read_file(seqDir + f"{frameNum}.png") -- or .jpg
# img = tf.image.decode_png(img, channels=4).numpy() -- or decode_jpeg
def imagenet_train_to_array():
    # root_folder = 'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\\train\\'
    # folders = [
    #     "n02219486", "n07753592", "n03792782", "n04285008", "n02123045", "n03063599",
    #     "n03180011", "n03272010", "n03481172", "n03544143", "n07734744", "n07873807",
    #     "n04141076", "n04120489"
    # ]

    root_folder = 'S:\Documents\Computational Visual Abstraction Project\datasets\imagenet_data\imagenet_object_localization_patched2019\ILSVRC\Data\CLS-LOC\\train\\'
    folders = os.listdir(root_folder)

    folder_label_list = ''

    train_data = []
    labels = []
    label = -1
    for folder in folders:
        label += 1
        print(label)
        folder_label_list = folder_label_list + '\n' + folder + " - " + str(label)
        for file in os.listdir(root_folder + folder + '\\'):
            img_path = root_folder + folder + '\\' + file
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = cv.resize(img.numpy(), dsize=(96, 96), interpolation=cv.INTER_AREA)
            img = tf.expand_dims(img, axis=0)
            img = keras.layers.Cropping2D(cropping=16)(img)[0].numpy()

            # cv.imshow('beep', img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            train_data.append(img)
            labels.append(label)

        if label == 127:
            break

    print(folder_label_list)

    np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\\64x64_128cat_imnet_subset\\train_x.npy', train_data)
    np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\\64x64_128cat_imnet_subset\\train_y.npy', labels)


def imagenet_sketch_to_array():
    root_folder = 'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\ImageNet-Sketch\sketch\\'
    folders = os.listdir(root_folder)

    folder_label_list = ''

    test_data = []
    labels = []
    label = -1
    for folder in folders:
        label += 1
        print(label)
        folder_label_list = folder_label_list + '\n' + folder + " - " + str(label)
        for file in os.listdir(root_folder + folder + '\\'):
            img_path = root_folder + folder + '\\' + file
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = cv.resize(img.numpy(), dsize=(96, 96), interpolation=cv.INTER_AREA)
            img = tf.expand_dims(img, axis=0)
            img = keras.layers.Cropping2D(cropping=16)(img)[0].numpy()

            # cv.imshow('beep', img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            test_data.append(img)
            labels.append(label)

        if label == 127:
            break

    print(folder_label_list)

    np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\\64x64_128cat_imnet_subset\\sketch_test_x.npy', test_data)
    np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\\64x64_128cat_imnet_subset\\sketch_test_y.npy', labels)


def subset_test_to_array():
    root_folder = 'S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\\test\\'
    folders = os.listdir(root_folder)

    test_data = []
    labels = []
    label = -1
    for folder in folders:
        label += 1
        print(label)
        for file in os.listdir(root_folder + folder + '\\'):
            img_path = root_folder + folder + '\\' + file
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=4).numpy()
            img[:,:,0] = 255 - img[:, :, 3]
            img[:,:,1] = 255 - img[:, :, 3]
            img[:,:,2] = 255 - img[:, :, 3]
            img = np.delete(img, 3, 2)
            img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA)
            img = (255 * (img == 255)).astype('uint8')
            # cv2.imshow('beep', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            test_data.append(img)
            labels.append(label)

    np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\\test_x.npy', test_data)
    np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\sketch_imagenet_subset\\test_y.npy', labels)


# sketchCat = "tree"
# cats = ""
# for i in range(2000):
#         if not os.path.isdir(f"sketch_data/converted_data/{sketchCat}/{sketchCat}_{i+1}/"):
#             cats = f"{cats} {i},"
#
# print(cats)

# prepareSketchTransferData()
# sketch_selector()
# imagenet_train_to_array()
# subset_test_to_array()
# trainsamps = np.load('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\\64x64_128cat_imnet_subset\\train_x.npy', allow_pickle=True)
# trainsamps = trainsamps.astype("float32")
# np.save('S:\Documents\Computational Visual Abstraction Project\datasets\sketch_data\\64x64_128cat_imnet_subset\\train_x_float32.npy', trainsamps)

# imagenet_sketch_to_array()