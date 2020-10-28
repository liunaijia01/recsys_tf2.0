import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

class Linear(tf.keras.layers.Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        return self.dense(inputs)


class DNN(tf.keras.layers.Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        outputs = self.dropout(x)
        return outputs


class FM(tf.keras.layers.Layer):
    def __init__(self, k=10, w_reg=1e-4, v_reg=1e-4):
        """
        FM
        :params k: A scalar. latent vector dimension
        """
        super(FM, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,), initializer=tf.zeros_initializer(), trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer='random_uniform', regularizer=l2(self.w_reg), trainable=True)
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k), initializer='random_uniform', regularizer=l2(self.v_reg), trainable=True)


    def call(self, inputs, **kwargs):
        first_order = self.w0 + tf.matmul(inputs, self.w)
        second_order = 0.5 * tf.reduce_sum(tf.pow(tf.matmul(inputs, self.v), 2) - tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)),
                                           axis=1,
                                           keepdims=True)
        return first_order + second_order





