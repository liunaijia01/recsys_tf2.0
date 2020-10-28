import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2



class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4, embed_reg=1e-4):
        super(FM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.feature_length = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) \
                              + sum([feat['feat_num'] for feat in self.dense_feature_columns])
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.embed_layers = {
            'embed_' + str(i): self.add_weight(
                name='embed_' + str(i),
                shape=(feat['feat_num'], feat['embed_dim']),
                initializer='random_uniform',
                regularizer=l2(embed_reg),
            )
            for i, feat in enumerate(self.sparse_feature_columns)
        }

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, self.feature_length),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs = tf.concat([inputs[feat['name']] for feat in self.dense_feature_columns], axis=1)
        sparse_embed_list = []
        for i, feat in enumerate(self.sparse_feature_columns):
            feat_value = inputs[feat['name']]
            weights = self.embed_layers['embed_{}'.format(i)]
            sparse_embed_list.append(tf.matmul(feat_value, weights))
        sparse_embed = tf.concat(sparse_embed_list, axis=1)
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1)
        first_order = self.w0 + tf.matmul(stack, self.w)
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(stack, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(stack, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)
        outputs = first_order + second_order
        return outputs


class FM(tf.keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs