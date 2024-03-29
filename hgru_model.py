import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import tensorflow_addons as tfa
import numpy as np
from typeguard import typechecked
import logging
import sketch_utils as utils
import os
import cv2 as cv

#   Disables GPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Types for typing functions signatures."""

from typing import Union, Callable, List

# TODO: Remove once https://github.com/tensorflow/tensorflow/issues/44613 is resolved
# from tensorflow.python.keras.engine import keras_tensor


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable
    # keras_tensor.KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.

    Source: "Group Normalization" (Yuxin Wu & Kaiming He, 2018)
    https://arxiv.org/abs/1803.08494

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.

    Args:
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
            Defaults to 32.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    @typechecked
    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: Initializer = "zeros",
        gamma_initializer: Initializer = "ones",
        beta_regularizer: Regularizer = None,
        gamma_regularizer: Regularizer = None,
        beta_constraint: Constraint = None,
        gamma_constraint: Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class InstanceNormalization(GroupNormalization):
    """Instance normalization layer.

    Instance Normalization is an specific case of ```GroupNormalization```since
    it normalizes all features of one channel. The Groupsize is equal to the
    channel size. Empirically, its accuracy is more stable than batch norm in a
    wide range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Arguments
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape
        Same shape as input.

    References
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
        (https://arxiv.org/abs/1607.08022)
    """

    def __init__(self, **kwargs):
        if "groups" in kwargs:
            logging.warning("The given value for groups will be overwritten.")

        kwargs["groups"] = -1
        super().__init__(**kwargs)


class WeightSymmetry(keras.constraints.Constraint):
    def __init__(self, name=None):
        self.ortho_init = keras.initializers.Orthogonal()
        self.prev_weights = None
        self.name = name
        # self.name = None


    def __call__(self, w):
        #   For initialized models that have already trained at least an epoch.
        # print(self.name)
        # if self.name is not None:
        #     self.prev_weights = tf.convert_to_tensor(np.load(
        #         f'symmetric_weights/rcnn_cand_10b_weights_3ep/{self.name}.npy',
        #         allow_pickle=True
        #     ))
        # print("Begin prev Pair")
        # print(self.prev_weights[0][0][0][1])
        # print(self.prev_weights[0][0][1][0])
        # print("End prev Pair")

        if self.prev_weights is None:
            #   Only run ortho init on first epoch!!!
            w = self.ortho_init(shape=w.shape)
            up = tf.linalg.band_part(w, 0, -1)
            w = up + tf.linalg.band_part(tf.transpose(up, perm=[0, 1, 3, 2]), -1, -1)

            self.prev_weights = w
        else:
            prev = tf.identity(self.prev_weights)
            grad = (prev - w) / 0.001
            grad = (grad + tf.transpose(grad, perm=[0, 1, 3, 2]))*0.5
            w = prev - (0.001 * grad)
            self.prev_weights = w
            print("Begin Pair")
            print(w[0][0][0][1])
            print(w[0][0][1][0])
            print("End Pair")

        return w


#   How to compute a frequency threshold?  Only blur if high frequencies are high enough (power threshold)
#   in kernel.
class SmoothKernel(keras.constraints.Constraint):
    def __call__(self, w):
        # w = tf.transpose(w, perm=[2, 0, 1, 3])
        # print(w.shape)
        # print(w)

        w = tf.transpose(w, perm=[2, 3, 0, 1])
        copy = w.numpy()
        # mean, var = tf.nn.moments(tf.reshape(w, shape=[w.shape[0] * w.shape[1], w.shape[2], w.shape[3]]), axes=[1, 2])
        mean, var = tf.nn.moments(w, axes=[2, 3])
        # print("before check")
        # print(tf.reduce_sum(var))

        #   Obtain indices from vars where vars[i] >= threshold.  If so, run these kernels through the blur.
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                v = var[i][j]
                # print(v)

                # kool = tf.expand_dims(w[i][j], axis=-1)
                # print(kool.shape)
                # print(kool)

                if v >= 0.2:
                    copy[i][j] = utils.gaussian_blur(tf.expand_dims(w[i][j], axis=-1), 3, 0.5)[0]
                elif v <= 0.02:
                    #   Visualizing filters revealed an inversion of color after Laplacian.
                    #   This is why I negate with -1 to hopefully correct that.
                    #   First blur out noise then apply laplacian.
                    copy[i][j] = utils.gaussian_blur(tf.expand_dims(w[i][j], axis=-1), 3, 0.5)[0]
                    copy[i][j] = -1 * cv.Laplacian(w[i][j].numpy(), cv.CV_32F, ksize=3, scale=0.25)
                else:
                    copy[i][j] = w[i][j]

        copy = tf.convert_to_tensor(copy)
        # mean, var = tf.nn.moments(copy, axes=[2, 3])
        # print("after check")
        # print(tf.reduce_sum(var))


        # for i in w:
        #     for k in i:
        #         # k = tf.reshape(k, shape=[k.shape[0] * k.shape[1]])
        #         # std = np.std(w[i][k])
        #         mean, var = tf.nn.moments(tf.reshape(k, shape=[k.shape[0] * k.shape[1]]), axes=[0])
        #         # print(var)



        # w = utils.gaussian_blur(tf.transpose(w, perm=[2, 0, 1, 3]), 3, 0.5)
        return tf.transpose(copy, perm=[2, 3, 0, 1])


class V1_Helper_Layer(layers.Layer):
    def __init__(self, out_channels, v, mult):
        super(V1_Helper_Layer, self).__init__()

        self.layer = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=v), trainable=True)
        self.mult = mult

    def call(self, inp, **kwargs):
        if self.mult:
            return self.layer * inp
        else:
            return self.layer + inp


class hGRU_V1(layers.Layer):
    def __init__(self, out_channels):
        super(hGRU_V1, self).__init__()

        self.hidden_state = None
        self.init_state = None

        self.nonlin = layers.Activation(activations.relu)
        self.norm = InstanceNormalization()

        self.Noise = layers.GaussianNoise(0.5)
        self.Blur = layers.Lambda(lambda x: utils.gaussian_blur(x, 3, 0.5))

        self.Add = layers.Add()
        self.Mult = layers.Multiply()
        self.Sub = layers.Subtract()

        self.conv_inp = layers.Conv2D(out_channels, kernel_size=7, strides=(1,1), padding="same", use_bias=False, kernel_constraint=SmoothKernel())
        # self.conv_lat = layers.Conv2D(out_channels, kernel_size=7, strides=(1,1), padding="same", use_bias=False)
        # self.conv_fb1 = layers.Conv2DTranspose(out_channels, kernel_size=7, strides=(2,2), padding="same", use_bias=False)
        # self.conv_fb2 = layers.Conv2DTranspose(out_channels, kernel_size=9, strides=(4,4), padding="same", use_bias=False)

        # self.block = [
        #     self.norm,
        #     self.nonlin,
        #     layers.Conv2D(out_channels, kernel_size=5, strides=(1,1), padding="same", use_bias=False),
        #     self.Blur,
        #     self.Noise,
        #     self.norm,
        #     self.nonlin
        # ]

        #   Serre Lab hGRU V1 materials.
        self.gru_t = 8
        hgru_kernel_size = 5
        self.u1_gate = layers.Conv2D(out_channels, kernel_size=1, padding="same",
                                     bias_initializer=keras.initializers.RandomUniform(1, 7),
                                     kernel_initializer=keras.initializers.Orthogonal())
        self.u2_gate = layers.Conv2D(out_channels, kernel_size=1, padding="same",
                                     kernel_initializer=keras.initializers.Orthogonal())
        self.u1_gate(tf.zeros(shape=(1, 1, 1, out_channels)))
        self.u2_gate(tf.zeros(shape=(1, 1, 1, out_channels)))
        self.u2_gate.set_weights([self.u2_gate.get_weights()[0], -self.u1_gate.get_weights()[1]])

        self.w_gate_inh = tf.Variable(initial_value=keras.initializers.Orthogonal()(shape=(hgru_kernel_size, hgru_kernel_size, out_channels, out_channels)),
                                      trainable=True)#, constraint=SmoothKernel())
        self.w_gate_exc = tf.Variable(initial_value=keras.initializers.Orthogonal()(shape=(hgru_kernel_size, hgru_kernel_size, out_channels, out_channels)),
                                      trainable=True)#, constraint=SmoothKernel())

        # self.alpha = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=0.1), trainable=True)
        # self.gamma = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=1.0), trainable=True)
        # self.kappa = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=0.5), trainable=True)
        # self.w = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=0.5), trainable=True)
        # self.mu = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=1.0), trainable=True)

        self.alpha = V1_Helper_Layer(out_channels, 0.1, True)
        self.gamma = V1_Helper_Layer(out_channels, 1.0, True)
        self.kappa = V1_Helper_Layer(out_channels, 0.5, True)
        self.w = V1_Helper_Layer(out_channels, 0.5, True)
        self.mu = V1_Helper_Layer(out_channels, 1.0, False)

        self.sigmoid = layers.Activation(activations.sigmoid)
        self.tanh = layers.Activation(activations.tanh)
        self.relu = layers.Activation(activations.relu)

        # self.norm = layers.Lambda(lambda x: tf.nn.local_response_normalization(x))

        for t in range(self.gru_t):
            setattr(self, f'gru_norm1_{t}', InstanceNormalization())
            setattr(self, f'gru_norm2_{t}', InstanceNormalization())
            setattr(self, f'gru_norm3_{t}', InstanceNormalization())
            setattr(self, f'gru_norm4_{t}', InstanceNormalization())
        #   End.


    def call(self, x, td_inp=None, td_inp2=None):

        if x is not None:
            x = self.conv_inp(x)
        else:
            # x = self.Add([self.init_state, self.conv_lat(self.hidden_state)])
            # x = self.init_state
            x = self.hidden_state

        if td_inp is not None:
            x = self.Add([x, self.conv_fb1(td_inp)])

        if td_inp2 is not None:
            x = self.Add([x, self.conv_fb2(td_inp2)])#, self.conv_lat(self.hidden_state)])
            # x = self.Noise(x)
            # x = self.norm(x)
            # x = self.relu(x)
            #
            # self.hidden_state = x
            # return self.hidden_state

        x = self.Blur(x)
        x = self.Noise(x)

        # for stage in self.block:
        #     x = stage(x)

        # if self.hidden_state is not None:
        #     if self.gru_t is 0:
        #         self.hidden_state = x
        #         return x
        #
        #     for t in range(self.gru_t):
        #         g1_t = self.sigmoid(getattr(self, f'gru_norm1_{t}')(self.u1_gate(self.hidden_state)))
        #         c1_t = getattr(self, f'gru_norm2_{t}')(tf.nn.conv2d(self.hidden_state * g1_t, self.w_gate_inh, padding="SAME", strides=(1,1)))
        #
        #         next_state1 = self.relu(x - self.relu(c1_t * (self.alpha * self.hidden_state + self.mu)))
        #
        #         g2_t = self.sigmoid(getattr(self, f'gru_norm3_{t}')(self.u2_gate(next_state1)))
        #         c2_t = getattr(self, f'gru_norm4_{t}')(tf.nn.conv2d(next_state1, self.w_gate_exc, padding="SAME", strides=(1,1)))
        #
        #         h2_t = self.relu(self.kappa * next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
        #
        #         self.hidden_state = (1 - g2_t) * self.hidden_state + g2_t * h2_t
        #
        #     return self.hidden_state
        # else:
        #     self.init_state = x
        #     self.hidden_state = x

        # if self.hidden_state is not None:
        #     if self.gru_t is 0:
        #         self.hidden_state = x
        #         return x
        #
        #     for t in range(self.gru_t):
        #         g1_t = self.sigmoid(getattr(self, f'gru_norm1_{t}')(self.u1_gate(self.hidden_state)))
        #         c1_t = getattr(self, f'gru_norm2_{t}')(tf.nn.conv2d(self.Mult([self.hidden_state, g1_t]), self.w_gate_inh, padding="SAME", strides=(1,1)))
        #
        #         next_state1 = self.relu(self.Sub([x, self.relu(self.Mult([c1_t, self.mu(self.alpha(self.hidden_state))]))]))
        #
        #         g2_t = self.sigmoid(getattr(self, f'gru_norm3_{t}')(self.u2_gate(next_state1)))
        #         c2_t = getattr(self, f'gru_norm4_{t}')(tf.nn.conv2d(next_state1, self.w_gate_exc, padding="SAME", strides=(1,1)))
        #
        #         h2_t = self.relu(self.Add([self.kappa(next_state1), self.gamma(c2_t), self.Mult([self.w(next_state1), c2_t])]))
        #
        #         self.hidden_state = self.Add([self.Mult([self.Sub([1.0, g2_t]), self.hidden_state]), self.Mult([g2_t, h2_t])])
        #
        #     # return self.hidden_state
        # else:
        #     self.init_state = x
        #     self.hidden_state = x

        #   Used in class 13 models.
        if self.init_state is None:
            x = tf.math.square(x)
            x = utils.gaussian_blur(x, 3, 0.5)
            x = self.Noise(x)
            # x = self.Add([x, tf.random.normal(x.shape, stddev=0.5)])
            self.init_state = x

            for t in range(self.gru_t):
                g1_t = self.sigmoid(getattr(self, f'gru_norm1_{t}')(self.u1_gate(self.init_state)))
                # g1_t = self.sigmoid(self.norm(self.u1_gate(self.init_state)))
                c1_t = getattr(self, f'gru_norm2_{t}')(tf.nn.conv2d(self.init_state * g1_t, self.w_gate_inh, padding="SAME", strides=(1,1)))
                # c1_t = self.norm(tf.nn.conv2d(self.init_state * g1_t, self.w_gate_inh, padding="SAME", strides=(1,1)))

                next_state1 = self.relu(x - self.relu(c1_t * self.mu(self.alpha(self.init_state))))

                g2_t = self.sigmoid(getattr(self, f'gru_norm3_{t}')(self.u2_gate(next_state1)))
                # g2_t = self.sigmoid(self.norm(self.u2_gate(next_state1)))
                c2_t = getattr(self, f'gru_norm4_{t}')(tf.nn.conv2d(next_state1, self.w_gate_exc, padding="SAME", strides=(1,1)))
                # c2_t = self.norm(tf.nn.conv2d(next_state1, self.w_gate_exc, padding="SAME", strides=(1,1)))

                h2_t = self.relu(self.kappa(next_state1) + self.gamma(c2_t) + self.w(next_state1)*c2_t)

                self.init_state = (1 - g2_t) * self.init_state + g2_t * h2_t

            self.init_state = utils.gaussian_blur(self.init_state, 3, 0.5)
            self.hidden_state = self.Noise(self.init_state)
            # self.hidden_state = self.Add([self.init_state, tf.random.normal(self.init_state.shape, stddev=0.5)])
            # self.hidden_state = self.init_state
        else:
            self.hidden_state = x

        return self.hidden_state
        # return x


#   Based on CORnet-S.
#   Source: https://github.com/dicarlolab/CORnet
class RecurrentCNNBlock(layers.Layer):

    scale = 4

    def __init__(self, out_channels, times=1, kernel_size=3, smooth_kernel=False, use_fb=False, us_factor=None, pool_factor=None):
        super(RecurrentCNNBlock, self).__init__()

        self.hidden_state = None

        self.times = times
        self.use_fb = use_fb

        self.conv_inp = layers.Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)
        # self.conv_lat = layers.Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)

        if us_factor is not None:
            self.conv_fb = layers.Conv2DTranspose(out_channels, kernel_size=1, strides=(us_factor, us_factor), padding="same", use_bias=False)

        if pool_factor is not None:
            self.conv_ff_skip = layers.Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)
            self.skip_pool = layers.MaxPooling2D(pool_size=(pool_factor, pool_factor), strides=(pool_factor, pool_factor), padding="same")

        self.skip = layers.Conv2D(out_channels, kernel_size=1, strides=(1,1), padding="same", use_bias=False)
        self.norm_skip = InstanceNormalization()
        # self.norm = layers.Lambda(lambda x: tf.nn.local_response_normalization(x))

        if smooth_kernel:
            self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=kernel_size, padding="same", use_bias=False, kernel_constraint=SmoothKernel())
            self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=kernel_size, strides=(1,1), padding="same", use_bias=False, kernel_constraint=SmoothKernel())
            self.conv3 = layers.Conv2D(out_channels, kernel_size=kernel_size, padding="same", use_bias=False, kernel_constraint=SmoothKernel())
        else:
            self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=kernel_size, padding="same", use_bias=False)
            self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=kernel_size, strides=(1,1), padding="same", use_bias=False)
            self.conv3 = layers.Conv2D(out_channels, kernel_size=kernel_size, padding="same", use_bias=False)

        self.nonlin2 = layers.Activation(activations.relu)
        self.nonlin1 = layers.Activation(activations.relu)
        self.nonlin3 = layers.Activation(activations.relu)

        self.norm_fb = InstanceNormalization()
        self.Add = layers.Add()
        self.Noise = layers.GaussianNoise(0.5)
        self.Blur = layers.Lambda(lambda x: utils.gaussian_blur(x, 3, 0.5))

        for t in range(self.times):
            setattr(self, f'norm1_{t}', InstanceNormalization())
            setattr(self, f'norm2_{t}', InstanceNormalization())
            setattr(self, f'norm3_{t}', InstanceNormalization())

        #   Serre Lab hGRU material here.
        # self.u1_gate = layers.Conv2D(out_channels, kernel_size=1, padding="same",
        #                              bias_initializer=keras.initializers.RandomUniform(1, 7))
        # self.u2_gate = layers.Conv2D(out_channels, kernel_size=1, padding="same")
        # self.u1_gate(tf.zeros(shape=(1, 1, 1, out_channels)))
        # self.u2_gate(tf.zeros(shape=(1, 1, 1, out_channels)))
        # self.u2_gate.set_weights([self.u2_gate.get_weights()[0], -self.u1_gate.get_weights()[1]])
        #
        # self.w_gate_inh = tf.Variable(initial_value=keras.initializers.GlorotUniform()(shape=(3, 3, out_channels, out_channels)),
        #                               trainable=True, constraint=WeightSymmetry())
        # self.w_gate_exc = tf.Variable(initial_value=keras.initializers.GlorotUniform()(shape=(3, 3, out_channels, out_channels)),
        #                               trainable=True, constraint=WeightSymmetry())
        #
        # self.alpha = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=0.1), trainable=True)
        # self.gamma = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=1.0), trainable=True)
        # self.kappa = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=0.5), trainable=True)
        # self.w = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=0.5), trainable=True)
        # self.mu = tf.Variable(initial_value=tf.fill([1, 1, out_channels], value=1.0), trainable=True)
        #
        # self.gru_t = gru_t
        # self.sigmoid = layers.Activation(activations.sigmoid)
        # self.tanh = layers.Activation(activations.tanh)
        # self.relu = layers.Activation(activations.relu)
        #
        # for t in range(self.gru_t):
        #     setattr(self, f'gru_norm1_{t}', layers.BatchNormalization(momentum=0.1))
        #     setattr(self, f'gru_norm2_{t}', layers.BatchNormalization(momentum=0.1))
        #     setattr(self, f'gru_norm3_{t}', layers.BatchNormalization(momentum=0.1))
        #     setattr(self, f'gru_norm4_{t}', layers.BatchNormalization(momentum=0.1))


    #   Recurrent CNN call.
    #   Sourced:  https://github.com/cjspoerer/rcnn-sat
    def call(self, inp, td_inp=None, skip_inp=None, fb=False, fb2=False, training=False):

        if self.hidden_state is None:
            x = self.conv_inp(inp)
        else:
            x = self.hidden_state

            # if skip_inp is not None:
            #     x = self.Add([x, self.skip_pool(self.conv_ff_skip(skip_inp))])
        # else:
            # if skip_inp is not None:
            #     x = self.Add([self.conv_lat(self.hidden_state), self.skip_pool(self.conv_ff_skip(skip_inp))])
            # else:
            #     x = self.conv_lat(self.hidden_state)
            #
            # x = self.Noise(x)
            # x = self.norm_fb(x)
            # x = self.nonlin1(x)
            # self.hidden_state = x
            # return x

        if skip_inp is not None:
            x = self.Add([x, self.skip_pool(self.conv_ff_skip(skip_inp))])

        if td_inp is not None:
            x = self.Add([x, self.conv_fb(td_inp)])

        x = self.Blur(x)
        x = self.Noise(x)
        # x = self.Add([x, tf.random.normal(x.shape, stddev=0.5)])

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                # skip = self.norm(self.skip(x))
                self.conv2.strides = (1,1)
            else:
                skip = x
                self.conv2.strides = (1,1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            # x = self.norm(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            # x = self.norm(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            # x = self.norm(x)
            x = self.Add([x, skip])

            x = self.nonlin3(x)

        # x = self.Noise(x)

        #   Serre Lab hGRU computations.
        # if self.hidden_state is not None:
        #     for t in range(self.gru_t):
        #         g1_t = self.sigmoid(getattr(self, f'gru_norm1_{t}')(self.u1_gate(self.hidden_state)))
        #         c1_t = getattr(self, f'gru_norm2_{t}')(tf.nn.conv2d(self.hidden_state * g1_t, self.w_gate_inh, padding="SAME", strides=(1,1)))
        #
        #         next_state1 = self.relu(x - self.relu(c1_t * (self.alpha * self.hidden_state + self.mu)))
        #
        #         g2_t = self.sigmoid(getattr(self, f'gru_norm3_{t}')(self.u2_gate(next_state1)))
        #         c2_t = getattr(self, f'gru_norm4_{t}')(tf.nn.conv2d(next_state1, self.w_gate_exc, padding="SAME", strides=(1,1)))
        #
        #         h2_t = self.relu(self.kappa * next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
        #
        #         self.hidden_state = (1 - g2_t) * self.hidden_state + g2_t * h2_t
        # else:
        #     self.hidden_state = x

        # x = self.Add([x, tf.random.normal(x.shape, stddev=0.5)])
        self.hidden_state = x

        return x


class RecurrentCNN(keras.Model):
    def __init__(self, fb_loops=0):
        super(RecurrentCNN, self).__init__()

        self.fb_loops = fb_loops

        self.V1 = hGRU_V1(64)
        self.mp = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        # self.mp = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        # self.V2 = RecurrentCNNBlock(128, times=1, use_fb=True, us_factor=2)
        self.V2 = RecurrentCNNBlock(128, times=2, kernel_size=5, smooth_kernel=False)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        # self.V4 = RecurrentCNNBlock(256, times=1, use_fb=True, us_factor=2, pool_factor=8)
        self.V4 = RecurrentCNNBlock(256, times=4, kernel_size=3, smooth_kernel=False, pool_factor=4)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.IT = RecurrentCNNBlock(512, times=2, kernel_size=3, use_fb=False)

        self.V1_Lam = layers.Lambda(lambda x: x, name="V1")
        self.V2_Lam = layers.Lambda(lambda x: x, name="V2")
        self.V4_Lam = layers.Lambda(lambda x: x, name="V4")
        self.IT_Lam = layers.Lambda(lambda x: x, name="IT")

        self.decoder = [
            layers.GlobalAveragePooling2D(),
            # tfa.layers.AdaptiveAveragePooling2D(1),
            # layers.Flatten(),
            layers.Dense(9, activation='softmax')
        ]


    #   Recurrent CNN call.
    def call(self, x, training=None, mask=None):

        x = self.V1(x)
        x = self.mp(x)
        x = self.V2(x)
        x = self.mp1(x)
        x = self.V4(x, skip_inp=self.V1.hidden_state)
        x = self.mp2(x)
        x = self.IT(x)

        # for t in range(self.fb_loops-1):
        #     # x = self.V1(None, td_inp=self.V2.hidden_state, td_inp2=self.V4.hidden_state)
        #     x = self.V1(None, td_inp2=self.V4.hidden_state)
        #     x = self.mp(x)
        #
        #     # x = self.V2(x, td_inp=self.V4.hidden_state)
        #     x = self.V2(x)
        #     x = self.mp1(x)
        #
        #     # x = self.V4(x, td_inp=self.IT.hidden_state, skip_inp=self.V1.hidden_state)
        #     x = self.V4(x, skip_inp=self.V1.hidden_state)
        #     x = self.mp2(x)
        #
        #     self.IT(x)
        #
        # # x = self.V1(None, td_inp=self.V2.hidden_state, td_inp2=self.V4.hidden_state)
        # # x = self.V1(None)
        # x = self.V1(None, td_inp2=self.V4.hidden_state)
        # x = self.mp(x)
        # x = self.V1_Lam(x)
        #
        # # x = self.V2(x, td_inp=self.V4.hidden_state)
        # x = self.V2(x)
        # x = self.mp1(x)
        # x = self.V2_Lam(x)
        #
        # # x = self.V4(x, td_inp=self.IT.hidden_state, skip_inp=self.V1.hidden_state)
        # x = self.V4(x, skip_inp=self.V1.hidden_state)
        # x = self.mp2(x)
        # x = self.V4_Lam(x)
        #
        # x = self.IT(x)
        # x = self.IT_Lam(x)

        #   Reset all hidden states to None here.
        self.V1.hidden_state = None
        self.V1.init_state = None
        self.V2.hidden_state = None
        self.V4.hidden_state = None
        self.IT.hidden_state = None

        for stage in self.decoder:
            x = stage(x)

        return x


# model = RecurrentCNN(fb_loops=4)
# inputs = keras.Input(shape=(64,64,3))
# outputs = model(inputs)
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(lr=0.001),
#     metrics=["accuracy"],
#     # run_eagerly=True
# )
#
# model.load_weights('trained_sketch_rec_models/SketchTransferCandidates/rcnn_cand_10e_weights_1ep/')
#
# # print(model.summary())
# # print(model.get_layer("recurrent_cnn").get_layer("h_gru__v1").w_gate_inh)
#
# w = model.get_layer("recurrent_cnn").get_layer("h_gru__v1").mu.layer
# print(w)

# tf.io.write_file('trained_sketch_rec_models/SketchTransferCandidates/symmetric_weights/rcnn_cand_10b_weights_3ep/w_gate_inh',
#                  tf.strings.as_string(model.get_layer("recurrent_cnn").get_layer("h_gru__v1").w_gate_inh))
#
# tf.io.write_file('trained_sketch_rec_models/SketchTransferCandidates/symmetric_weights/rcnn_cand_10b_weights_3ep/w_gate_exc',
#                  tf.strings.as_string(model.get_layer("recurrent_cnn").get_layer("h_gru__v1").w_gate_exc))

# np.save('trained_sketch_rec_models/SketchTransferCandidates/symmetric_weights/rcnn_cand_10b_weights_3ep/w_gate_inh',
#         model.get_layer("recurrent_cnn").get_layer("h_gru__v1").w_gate_inh.numpy())
#
# np.save('trained_sketch_rec_models/SketchTransferCandidates/symmetric_weights/rcnn_cand_10b_weights_3ep/w_gate_exc',
#         model.get_layer("recurrent_cnn").get_layer("h_gru__v1").w_gate_exc.numpy())