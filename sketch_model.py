import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers
from tensorflow.python.keras.layers import \
    SimpleRNNCell
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
from tensorflow.python.ops import array_ops
import numpy as np

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
class ConvLSTMModel(keras.Model):
    def __init__(self, seq_length=8):
        super(ConvLSTMModel, self).__init__()
        #   Model 23 revised.
        # self.l1 = ConvRNN2D(ConvLSTM2DCellNormed(128, 7, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.mp1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        # self.l2 = ConvRNN2D(ConvLSTM2DCellNormed(256, 5, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.mp2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        # self.l3 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=False, return_state=True)

        #   Model 20/21.
        # self.l1 = MaskConvLSTM2D(256, return_seq=True, kernel_size=3)
        # self.l2 = MaskConvLSTM2D(256, return_seq=False, kernel_size=3)
        # self.l3 = MaskConvLSTM2D(256, return_seq=False, kernel_size=3)

        # #   Model 26.
        # self.l1 = ConvRNN2D(ConvLSTM2DCellNormed(64, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l2 = ConvRNN2D(ConvLSTM2DCellNormed(64, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l3 = ConvRNN2D(ConvLSTM2DCellNormed(64, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.mp1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        # self.l4 = ConvRNN2D(ConvLSTM2DCellNormed(128, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l5 = ConvRNN2D(ConvLSTM2DCellNormed(128, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l6 = ConvRNN2D(ConvLSTM2DCellNormed(128, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.mp2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        # self.l7 = ConvRNN2D(ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l8 = ConvRNN2D(ConvLSTM2DCellNormed(256, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.mp3 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        self.l1 = ConvRNN2D(ConvLSTM2DCellNormed(128, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        self.l2 = ConvRNN2D(ConvLSTM2DCellNormed(128, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        self.mp1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        self.l3 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        self.mp2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        self.l4 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        self.mp3 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        self.l5 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=False, return_state=True)

        # self.mp4 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')
        # self.l13 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l14 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=True, return_state=True)
        # self.l15 = ConvRNN2D(ConvLSTM2DCellNormed(512, 3, padding='same', data_format="channels_last"), return_sequences=False, return_state=True)

        self.classifier = layers.Dense(10, activation='softmax')
        self.seq_length = seq_length
        self.lstm_layers = [self.l1, self.l2, self.mp1, self.l3, self.mp2, self.l4, self.mp3, self.l5]#, self.mp2, self.l5, self.l6]#, self.mp2, self.l7, self.l8, self.mp3, self.l9, self.l10]#, self.l11, self.l12]#, self.mp4, self.l13, self.l14, self.l15]

    #   Looping call.
    #   Need to loop single image n times with the norm occurring after each pass.
    # def call(self, input_tensor, training=None, mask=None):
    #     hidden_states = []
    #     for i in range(self.seq_length):
    #         x = None
    #         for j in range(len(self.lstm_layers)):
    #             l = self.lstm_layers[j]
    #             if i is 0:
    #                 if l is self.l1:
    #                     x = l(input_tensor)
    #                 else:
    #                     x = l(x[0])
    #             else:
    #                 if l is self.l1:
    #                     x = l(input_tensor, initial_state=[hidden_states[j][1], hidden_states[j][2]])
    #                 else:
    #                     x = l(x[0], initial_state=[hidden_states[j][1], hidden_states[j][2]])
    #
    #             if i is 0:
    #                 hidden_states.append(x)
    #             else:
    #                 hidden_states[j] = x
    #
    #            # if j < len(self.mp_layers):
    #            #     x[0] = self.mp_layers[j](x[0])
    #
    #     out = layers.GlobalAveragePooling2D()(hidden_states[len(hidden_states)-1][0])
    #     return self.classifier(out)

    def call(self, input_tensor, training=None, mask=None):
        x = None
        for j in range(len(self.lstm_layers)):
            l = self.lstm_layers[j]
            if l is self.l1:
                x = l(input_tensor)
            #   elif l is a max pool
            elif l is self.mp1 or l is self.mp2 or l is self.mp3:# or l is self.mp4:
                x[0] = l(x[0])
            else:
                x = l(x[0])

        out = layers.GlobalAveragePooling2D()(x[0])
        return self.classifier(out)

    # def model(self):
    #     x = keras.Input(shape=(8, 32, 32, 3))
    #     return keras.Model(inputs=[x], outputs=self.call(x))


class ConvLSTM2DCellNormed(ConvLSTM2DCell):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.norm_layer = tfa.layers.InstanceNormalization()
        # self.norm_layer = layers.LayerNormalization()

    #   Copied from ConvLSTM2DCell call in convolutional_recurrent.py.  Only modification is the application of the normalization layer.
    def call(self, inputs, states, training=None):
        # outputs, new_states = super().call(inputs, states, training)

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if 0 < self.dropout < 1.:
          inputs_i = inputs * dp_mask[0]
          inputs_f = inputs * dp_mask[1]
          inputs_c = inputs * dp_mask[2]
          inputs_o = inputs * dp_mask[3]
        else:
          inputs_i = inputs
          inputs_f = inputs
          inputs_c = inputs
          inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
          h_tm1_i = h_tm1 * rec_dp_mask[0]
          h_tm1_f = h_tm1 * rec_dp_mask[1]
          h_tm1_c = h_tm1 * rec_dp_mask[2]
          h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
          h_tm1_i = h_tm1
          h_tm1_f = h_tm1
          h_tm1_c = h_tm1
          h_tm1_o = h_tm1

        (kernel_i, kernel_f,
         kernel_c, kernel_o) = array_ops.split(self.kernel, 4, axis=3)
        (recurrent_kernel_i,
         recurrent_kernel_f,
         recurrent_kernel_c,
         recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)

        if self.use_bias:
          bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, 4)
        else:
          bias_i, bias_f, bias_c, bias_o = None, None, None, None

        x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        #   Next, try normalization before recurrent activations too.  Then maybe only before recurrent and remove those before activation.
        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)

        #   Added normalization before activation.
        normed_xh = self.norm_layer(x_c + h_c)
        c = f * c_tm1 + i * self.activation(normed_xh)
        normed_c = self.norm_layer(c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(normed_c)

        return h, [h, c]