#   PyTorch to Keras CORnet-S translation.
#   Author:  André Longon
#   Root Source:  https://github.com/dicarlolab/CORnet
#   Source Code:  https://github.com/dicarlolab/CORnet/blob/master/cornet/cornet_s.py

from tensorflow import keras
from tensorflow.keras import layers, activations


class RecurrentCNNBlock(layers.Layer):

    scale = 4

    def __init__(self, out_channels, times=1):
        super(RecurrentCNNBlock, self).__init__()

        self.times = times

        self.conv_input = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.skip = layers.Conv2D(out_channels, kernel_size=1, strides=(2,2), use_bias=False)
        self.norm_skip = layers.BatchNormalization(epsilon=1e-05)

        self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=1, use_bias=False)
        self.nonlin1 = layers.Activation(activations.relu)
        self.padding = layers.ZeroPadding2D(padding=(1,1))
        self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=3, strides=(1,1), use_bias=False)
        self.pool = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.nonlin2 = layers.Activation(activations.relu)
        self.conv3 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.nonlin3 = layers.Activation(activations.relu)

        self.Add = layers.Add()

        for t in range(self.times):
            setattr(self, f'norm1_{t}', layers.BatchNormalization(epsilon=1e-05))
            setattr(self, f'norm2_{t}', layers.BatchNormalization(epsilon=1e-05))
            setattr(self, f'norm3_{t}', layers.BatchNormalization(epsilon=1e-05))


    def call(self, inp, td_inp=None, skip_inp=None, fb=False, fb2=False, training=False):

        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
            else:
                skip = x

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            #   Cannot change strides in keras I believe.
            #   Reassignment of stride parameter does not alter output shape.  Pooling as alternative approach.
            if t == 0:
                x = self.pool(x)

            x = self.padding(x)
            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x = self.Add([x, skip])
            x = self.nonlin3(x)

        return x


class RecurrentCNN(keras.Model):
    def __init__(self):
        super(RecurrentCNN, self).__init__()

        self.V1 = [
            layers.ZeroPadding2D(padding=(3,3)),
            layers.Conv2D(64, kernel_size=7, strides=(2,2), use_bias=False),
            layers.BatchNormalization(epsilon=1e-05),
            layers.Activation(activations.relu),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(64, kernel_size=3, strides=(1,1), use_bias=False),
            layers.BatchNormalization(epsilon=1e-05),
            layers.Activation(activations.relu)
        ]

        self.V2 = RecurrentCNNBlock(128, times=2)
        self.V4 = RecurrentCNNBlock(256, times=4)
        self.IT = RecurrentCNNBlock(512, times=2)

        self.decoder = [
            layers.GlobalAveragePooling2D(),
            layers.Dense(1000, activation='softmax')
        ]


    def call(self, x, training=None, mask=None):

        for stage in self.V1:
            x = stage(x)

        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)

        for stage in self.decoder:
            x = stage(x)

        return x