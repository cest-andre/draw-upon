import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa
from tensorflow.keras import layers, activations, regularizers
from tensorflow.python.keras.layers import \
    SimpleRNNCell
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
from tensorflow.python.ops import array_ops
import numpy as np
from typeguard import typechecked
import logging
import sketch_utils
# from tensorflow_addons.utils import types


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


#   Add a loop and hidden state feed procedure so that you can reuse a single image sequentially rather than copy paste.
#   Next, add a simplistic feedback procedure.
class MaskConvLSTM2D(layers.Layer):
    def __init__(self, numFilters, return_seq, kernel_size=3, return_state=False, data_format="channels_last", predict=False, **kwargs):
        super(MaskConvLSTM2D, self).__init__(**kwargs)
        self.return_seq = return_seq
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


#   Perform hidden state maintenance and sequence looping here.
#   Doesn't work because backprop through time doesn't respect my loop here.  It strictly follows the sequence length in the input_tensor.
#   I will need to get this working if I want to use glimpses to dynamically construct the sequence.
#   Try and use the cell defined below to manually step through sequence to see if BPTT will work.  I think it does!  We're learning!!
#   I should just initialize ConvRNN2D(ConvLSTM2DCellNormed()) layers like what is done in sketch_train.  Steps are performed in ConvRNN2D via the call method.
#
#   To test if bptt is working, train with old dataset and no outer loop with hidden state maintenance.
#   Then repeat again with new dataset and hs maint.  No norming.  See if timing and performance is significantly different.
#   I performed the above and feel reasonably confident that bptt does not cover all timesteps via manual looping.  I might need to write
#   a custom training function.
# class ConvLSTMModel(keras.Model):
#     def __init__(self, seq_length=8):
#         super(ConvLSTMModel, self).__init__()
#         #   Model 23 revised.
#         # self.l1 = ConvRNN2D(ConvLSTM2DCellNormed(128, 7, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
#         # self.mp1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
#         # self.l2 = ConvRNN2D(ConvLSTM2DCellNormed(256, 5, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
#         # self.mp2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
#         # self.l3 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=False, return_state=True)
#
#         #   Model 20/21.
#         # self.l1 = MaskConvLSTM2D(256, return_seq=True, kernel_size=3)
#         # self.l2 = MaskConvLSTM2D(256, return_seq=False, kernel_size=3)
#         # self.l3 = MaskConvLSTM2D(256, return_seq=False, kernel_size=3)
#
#         self.l1 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer=None, dropout=0.0, recurrent_dropout=0.0),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.l2 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer=None, dropout=0.0, recurrent_dropout=0.0),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.mp1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
#         self.l3 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer="l2", dropout=0.0, recurrent_dropout=0.5),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.l4 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer=None, dropout=0.0, recurrent_dropout=0.0),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.mp2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
#         self.l5 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer="l2", dropout=0.0, recurrent_dropout=0.5),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.l6 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer=None, dropout=0.0, recurrent_dropout=0.0),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.mp3 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
#         self.l7 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer="l2", dropout=0.0, recurrent_dropout=0.5),
#             return_sequences=True, return_state=True, activity_regularizer=None
#         )
#         self.l8 = ConvRNN2D(
#             ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last", kernel_regularizer=None, recurrent_regularizer=None, dropout=0.0, recurrent_dropout=0.0),
#             return_sequences=False, return_state=True, activity_regularizer=None
#         )
#
#         # self.fc1 = layers.Dense(64, activation='relu')
#         self.classifier = layers.Dense(9, activation='softmax')
#         self.seq_length = seq_length
#         self.lstm_layers = [
#                                 self.l1, self.l2, self.mp1, self.l3, self.l4, self.mp2, self.l5, self.l6, self.mp3, self.l7, self.l8
#                            ]
#
#
#     #   Looping call.
#     #   Need to loop single image n times with the norm occurring after each pass.
#     # def call(self, input_tensor, training=None, mask=None):
#     #     hidden_states = []
#     #     for i in range(self.seq_length):
#     #         x = None
#     #         for j in range(len(self.lstm_layers)):
#     #             l = self.lstm_layers[j]
#     #             if i is 0:
#     #                 if l is self.l1:
#     #                     x = l(input_tensor)
#     #                 else:
#     #                     x = l(x[0])
#     #             else:
#     #                 if l is self.l1:
#     #                     x = l(input_tensor, initial_state=[hidden_states[j][1], hidden_states[j][2]])
#     #                 else:
#     #                     x = l(x[0], initial_state=[hidden_states[j][1], hidden_states[j][2]])
#     #
#     #             if i is 0:
#     #                 hidden_states.append(x)
#     #             else:
#     #                 hidden_states[j] = x
#     #
#     #            # if j < len(self.mp_layers):
#     #            #     x[0] = self.mp_layers[j](x[0])
#     #
#     #     out = layers.GlobalAveragePooling2D()(hidden_states[len(hidden_states)-1][0])
#     #     return self.classifier(out)
#
#
#     def call(self, input_tensor, training=None, mask=None):
#         x = None
#         for j in range(len(self.lstm_layers)):
#             l = self.lstm_layers[j]
#             if l is self.l1:
#                 x = l(input_tensor)
#             #   elif l is a max pool
#             elif l is self.mp1 or l is self.mp2 or l is self.mp3:# or l is self.mp4:
#                 x[0] = l(x[0])
#             else:
#                 x = l(x[0])
#
#         out = layers.GlobalAveragePooling2D()(x[0])
#         # out = self.fc1(out)
#
#         return self.classifier(out)


    # def model(self):
    #     x = keras.Input(shape=(8, 32, 32, 3))
    #     return keras.Model(inputs=[x], outputs=self.call(x))


# #   Copied from ConvLSTM2DCell call in convolutional_recurrent.py.  Only modification is the application of the normalization layer.
# class ConvLSTM2DCellNormed(ConvLSTM2DCell):
#     def __init__(self, filters, kernel_size, **kwargs):
#         super().__init__(filters, kernel_size, **kwargs)
#         self.norm_layer = tfa.layers.InstanceNormalization()
#         # self.norm_layer = layers.LayerNormalization()
#
#
#     def call(self, inputs, states, training=None):
#         # outputs, new_states = super().call(inputs, states, training)
#
#         h_tm1 = states[0]  # previous memory state
#         c_tm1 = states[1]  # previous carry state
#
#         # dropout matrices for input units
#         dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
#         # dropout matrices for recurrent units
#         rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
#             h_tm1, training, count=4)
#
#         if 0 < self.dropout < 1.:
#           inputs_i = inputs * dp_mask[0]
#           inputs_f = inputs * dp_mask[1]
#           inputs_c = inputs * dp_mask[2]
#           inputs_o = inputs * dp_mask[3]
#         else:
#           inputs_i = inputs
#           inputs_f = inputs
#           inputs_c = inputs
#           inputs_o = inputs
#
#         if 0 < self.recurrent_dropout < 1.:
#           h_tm1_i = h_tm1 * rec_dp_mask[0]
#           h_tm1_f = h_tm1 * rec_dp_mask[1]
#           h_tm1_c = h_tm1 * rec_dp_mask[2]
#           h_tm1_o = h_tm1 * rec_dp_mask[3]
#         else:
#           h_tm1_i = h_tm1
#           h_tm1_f = h_tm1
#           h_tm1_c = h_tm1
#           h_tm1_o = h_tm1
#
#         (kernel_i, kernel_f,
#          kernel_c, kernel_o) = array_ops.split(self.kernel, 4, axis=3)
#         (recurrent_kernel_i,
#          recurrent_kernel_f,
#          recurrent_kernel_c,
#          recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)
#
#         if self.use_bias:
#           bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, 4)
#         else:
#           bias_i, bias_f, bias_c, bias_o = None, None, None, None
#
#         x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
#         x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
#         x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
#         x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)
#         h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
#         h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
#         h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
#         h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)
#
#         #   Next, try normalization before recurrent activations too.  Then maybe only before recurrent and remove those before activation.
#         i = self.recurrent_activation(x_i + h_i)
#         f = self.recurrent_activation(x_f + h_f)
#
#         #   Added normalization before activation.
#         normed_xh = self.norm_layer(x_c + h_c)
#         c = f * c_tm1 + i * self.activation(normed_xh)
#         normed_c = self.norm_layer(c)
#         o = self.recurrent_activation(x_o + h_o)
#         h = o * self.activation(normed_c)
#
#         return h, [h, c]


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


#   Based on CORnet-S.
#   Source: https://github.com/dicarlolab/CORnet
class RecurrentCNNBlock(layers.Layer):

    scale = 2

    def __init__(self, out_channels, times=1, use_fb=False, fb_channels=None, us_factor=None, pool_factor=None):
        super(RecurrentCNNBlock, self).__init__()

        self.hidden_state = None

        self.times = times
        self.use_fb = use_fb
        self.conv_input = layers.Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)
        self.skip = layers.Conv2D(out_channels, kernel_size=1, strides=(1,1), padding="same", use_bias=False)
        self.norm_skip = InstanceNormalization()

        self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=1, padding="same", use_bias=False)
        self.nonlin1 = layers.Activation(activations.relu)
        self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.nonlin2 = layers.Activation(activations.relu)
        self.conv3 = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
        self.nonlin3 = layers.Activation(activations.relu)

        if use_fb:
            # self.upsample = layers.UpSampling2D(size=(us_factor, us_factor))
            self.conv_fb = layers.Conv2DTranspose(fb_channels, kernel_size=3, strides=(us_factor,us_factor), padding="same", use_bias=False)

        if pool_factor is not None:
            self.conv_ff_skip = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
            self.skip_pool = layers.MaxPooling2D(pool_size=(pool_factor, pool_factor), strides=(pool_factor, pool_factor), padding="same")

        self.norm_fb = InstanceNormalization()
        self.conv_lat = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
        self.Add = layers.Add()

        # if use_fb2:
        #     self.conv_fb2 = layers.Conv2DTranspose(fb2_channels, kernel_size=3, strides=(us2_factor,us2_factor), padding="same", use_bias=False)
        #     self.norm_fb2 = tfa.layers.InstanceNormalization()

        for t in range(self.times):
            setattr(self, f'norm1_{t}', InstanceNormalization())
            setattr(self, f'norm2_{t}', InstanceNormalization())
            setattr(self, f'norm3_{t}', InstanceNormalization())

        self.Noise = layers.GaussianNoise(0.1)

        # self.norm = layers.Lambda(lambda x: tf.nn.local_response_normalization(x, alpha=.0001))

    #   CORNet call.
    # def call(self, inp, fb=False, fb2=False, training=False):
    #     x = self.conv_input(inp)
    #
    #     for t in range(self.times):
    #         if t == 0:
    #             skip = self.norm_skip(self.skip(x))
    #             self.conv2.strides = (1,1)
    #         else:
    #             skip = x
    #             self.conv2.strides = (1,1)
    #
    #         x = self.conv1(x)
    #         x = getattr(self, f'norm1_{t}')(x)
    #         x = self.nonlin1(x)
    #
    #         x = self.conv2(x)
    #         x = getattr(self, f'norm2_{t}')(x)
    #         x = self.nonlin2(x)
    #
    #         #   Define hidden_state in init and save after third nonlin if fb is true.  Then compute conv and norm for feedback.
    #         #   Next, when on the final intra-layer timestep but before third nonlin, if hidden_state is not None, feedback has occurred.
    #         #   This means the hidden state must be updated, so add x to hidden_state and relu, which updates hidden_state.  Next, determine
    #         #   if to feedback again or to pass hidden_state forward.
    #         # if t == self.times-1 and fb:
    #         #     # x = self.upsample(x)
    #         #     x = self.conv_fb(x)
    #         #     x = self.norm_fb(x)
    #         # else:
    #         x = self.conv3(x)
    #         x = getattr(self, f'norm3_{t}')(x)
    #         x += skip
    #
    #         if t != self.times-1:
    #             x = self.nonlin3(x)
    #
    #     #   Removing hidden state maintenance for now.
    #     if self.hidden_state is None:
    #         # x = self.nonlin3(x)
    #         self.hidden_state = x
    #     else:
    #         self.hidden_state += x
    #         # self.hidden_state = self.nonlin3(self.hidden_state)
    #         x = self.hidden_state
    #
    #     #   Rather than passing x into fb operations, pass hidden_state.
    #     if fb:
    #         x = self.conv_fb(x)
    #         x = self.norm_fb(x)
    #         # x = self.nonlin3(x)
    #     elif fb2:
    #         x = self.conv_fb2(x)
    #         x = self.norm_fb2(x)
    #         # x = self.nonlin3(x)
    #
    #     return self.nonlin3(x)

    #   Recurrent CNN call.
    #   Sourced:  https://github.com/cjspoerer/rcnn-sat
    def call(self, inp, td_inp=None, skip_inp=None, fb=False, fb2=False, training=False):

        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                # skip = self.skip(x)
                self.conv2.strides = (1,1)
            else:
                skip = x
                self.conv2.strides = (1,1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            # x = self.norm(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            # x = self.Noise(x)
            x = getattr(self, f'norm2_{t}')(x)
            # x = self.norm(x)
            x = self.nonlin2(x)

            #   Define hidden_state in init and save after third nonlin if fb is true.  Then compute conv and norm for feedback.
            #   Next, when on the final intra-layer timestep but before third nonlin, if hidden_state is not None, feedback has occurred.
            #   This means the hidden state must be updated, so add x to hidden_state and relu, which updates hidden_state.  Next, determine
            #   if to feedback again or to pass hidden_state forward.
            # if t == self.times-1 and fb:
            #     # x = self.upsample(x)
            #     x = self.conv_fb(x)
            #     x = self.norm_fb(x)
            # else:
            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            # x = self.norm(x)
            x = self.Add([x, skip])

            x = self.nonlin3(x)

        #   First step of recurrence so no top-down or lateral input yet.
        if self.hidden_state is None:
            x = self.Noise(x)
            self.hidden_state = x
        else:
            # x += self.conv_lat(self.hidden_state)
            x = self.Add([x, self.conv_lat(self.hidden_state)])
            if td_inp is not None:
                # x += self.conv_fb(td_inp)
                x = self.Add([x, self.conv_fb(td_inp)])

            if skip_inp is not None:
                x = self.Add([x, self.skip_pool(self.conv_ff_skip(skip_inp))])

            # x = self.Noise(x)

            x = self.norm_fb(x)
            x = self.nonlin3(x)
            x = self.Noise(x)
            self.hidden_state = x

        #   Rather than passing x into fb operations, pass hidden_state.
        # if fb:
        #     x = self.conv_fb(x)
        #     x = self.norm_fb(x)
        #     # x = self.nonlin3(x)
        # elif fb2:
        #     x = self.conv_fb2(x)
        #     x = self.norm_fb2(x)
        #     # x = self.nonlin3(x)

        return x


class RecurrentCNN(keras.Model):
    def __init__(self, fb_loops=0):
        super(RecurrentCNN, self).__init__()

        self.fb_loops = fb_loops

        self.V1 = [
            layers.Conv2D(64, kernel_size=7, strides=(1,1), padding="same", use_bias=False),
            InstanceNormalization(),
            layers.Activation(activations.relu),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
            layers.Conv2D(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False),
            InstanceNormalization(),
            layers.Activation(activations.relu),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        ]

        self.nonlin = layers.Activation(activations.relu)
        self.noise = layers.GaussianNoise(0.1)

        self.V1_Add = layers.Add()
        self.V1_conv_fb1 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.V1_conv_fb2 = layers.Conv2DTranspose(64, kernel_size=3, strides=(2,2), padding="same", use_bias=False)
        self.V1_conv_fb3 = layers.Conv2DTranspose(64, kernel_size=3, strides=(4,4), padding="same", use_bias=False)
        self.V1_norm_fb = InstanceNormalization()
        self.V1_conv_lat = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)

        self.V2 = RecurrentCNNBlock(128, times=2, use_fb=True, fb_channels=128, us_factor=2)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.V4 = RecurrentCNNBlock(256, times=4, use_fb=True, fb_channels=256, us_factor=2, pool_factor=2)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.IT = RecurrentCNNBlock(512, times=2, use_fb=False)

        self.V1_Lam = layers.Lambda(lambda x: x, name="V1")
        self.V2_Lam = layers.Lambda(lambda x: x, name="V2")
        self.V4_Lam = layers.Lambda(lambda x: x, name="V4")
        self.IT_Lam = layers.Lambda(lambda x: x, name="IT")

        self.decoder = [
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='softmax')
        ]

    # #   Single feedback connection call.
    # def call(self, x, training=None, mask=None):
    #
    #     for stage in self.V1:
    #         x = stage(x)
    #
    #     x = self.V2(x)
    #     x = self.mp1(x)
    #     x = self.V4(x)
    #     v4_state = x
    #
    #     for t in range(self.fb_loops):
    #         x = self.mp2(x)
    #
    #         if t == self.fb_loops-1:
    #             x = self.IT(x, fb=False)
    #         else:
    #             v4_state += self.IT(x, fb=True)
    #             v4_state = layers.Activation(activations.relu)(v4_state)
    #             x = v4_state
    #
    #     #   Reset all hidden states to None here.
    #     self.V2.hidden_state = None
    #     self.V4.hidden_state = None
    #     self.IT.hidden_state = None
    #
    #     for stage in self.decoder:
    #         x = stage(x)
    #
    #     return x

    #   Multi feedback connection call with hidden states.
    # def call(self, x, training=None, mask=None):
    #
    #     for stage in self.V1:
    #         x = stage(x)
    #
    #     #   First pass to save initial states and fb responses.
    #     v1_state = x
    #     x = layers.Activation(activations.relu)(x)
    #     x = self.V2(x, fb=False)
    #     x = self.mp1(x)
    #     x = self.V4(x, fb=False)
    #     x = self.mp2(x)
    #     x = self.IT(x, fb=False)
    #
    #
    #     for t in range(self.fb_loops):
    #         #   V2 -> V1 and feed through V2.
    #         v1_state += self.V2.norm_fb(self.V2.conv_fb(layers.Activation(activations.relu)(self.V2.hidden_state)))
    #         self.V2(layers.Activation(activations.relu)(v1_state), fb=False)
    #
    #         #   V4 -> V2 and feed through V4.
    #         self.V2.hidden_state += self.V4.norm_fb(self.V4.conv_fb(layers.Activation(activations.relu)(self.V4.hidden_state)))
    #         self.V4(self.mp1(layers.Activation(activations.relu)(self.V2.hidden_state)), fb=False)
    #
    #         #   IT -> V4 and feed through IT.
    #         self.V4.hidden_state += self.IT.norm_fb(self.IT.conv_fb(layers.Activation(activations.relu)(self.IT.hidden_state)))
    #         x = self.IT(self.mp2(layers.Activation(activations.relu)(self.V4.hidden_state)), fb=False)
    #
    #     #   Reset all hidden states to None here.
    #     self.V2.hidden_state = None
    #     self.V4.hidden_state = None
    #     self.IT.hidden_state = None
    #
    #     for stage in self.decoder:
    #         x = stage(x)
    #
    #     return x

    # #   Multi feedback connection call without hidden states.
    # def call(self, x, training=None, mask=None):
    #
    #     for stage in self.V1:
    #             x = stage(x)
    #
    #     for t in range(self.fb_loops):
    #         x = self.V2(x, fb=False)
    #         x = self.mp1(x)
    #
    #         #   V4 -> V2.
    #         x = self.V4(x, fb=True)
    #         x = self.V2(x, fb=False)
    #         x = self.mp1(x)
    #         x = self.V4(x, fb=False)
    #
    #         #   IT -> V4.
    #         x = self.IT(x, fb=True)
    #         x = self.V4(x, fb=False)
    #         x = self.mp2(x)
    #
    #         if t != self.fb_loops-1:
    #             x = self.IT(x, fb2=True)
    #
    #     x = self.IT(x, fb=False)
    #
    #     for stage in self.decoder:
    #         x = stage(x)
    #
    #     return x

    #   Recurrent CNN call.
    def call(self, x, training=None, mask=None):
        #   Remove 255 division in training setup.
        # x = keras.applications.vgg16.preprocess_input(x)

        # if x.shape[0] is not None:
        #     x = sketch_utils.augment_dataset(x)

        for stage in self.V1:
            x = stage(x)

        #   First pass to initialize states.
        v1_response = x
        v1_state = x
        x = self.noise(x)
        x = self.V2(x)
        x = self.mp1(x)
        x = self.V4(x)
        x = self.mp2(x)
        self.IT(x)

        for t in range(self.fb_loops-1):
            x = self.V1_Add([
                v1_response, self.V1_conv_lat(v1_state), self.V1_conv_fb1(self.V2.hidden_state),
                self.V1_conv_fb2(self.V4.hidden_state)
            ])

            x = self.V1_norm_fb(x)
            x = self.nonlin(x)
            x = self.noise(x)
            v1_state = x

            x = self.V2(x, td_inp=self.V4.hidden_state)
            x = self.mp1(x)

            x = self.V4(x, td_inp=self.IT.hidden_state, skip_inp=v1_state)
            x = self.mp2(x)

            self.IT(x)

        #   For final pass, include lambda identity layers in hopes that these final values are used rather than ones earlier.
        x = self.V1_Add([
            v1_response, self.V1_conv_lat(v1_state), self.V1_conv_fb1(self.V2.hidden_state),
            self.V1_conv_fb2(self.V4.hidden_state)
        ])

        x = self.V1_norm_fb(x)
        x = self.nonlin(x)
        x = self.noise(x)
        v1_state = x
        x = self.V1_Lam(x)

        x = self.V2(x, td_inp=self.V4.hidden_state)
        x = self.mp1(x)
        x = self.V2_Lam(x)

        x = self.V4(x, td_inp=self.IT.hidden_state, skip_inp=v1_state)
        x = self.mp2(x)
        x = self.V4_Lam(x)

        x = self.IT(x)
        x = self.IT_Lam(x)

        #   Reset all hidden states to None here.
        self.V2.hidden_state = None
        self.V4.hidden_state = None
        self.IT.hidden_state = None

        for stage in self.decoder:
            x = stage(x)

        return x







