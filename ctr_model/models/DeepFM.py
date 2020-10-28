import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
from ctr_model.layers.layers import FM, DNN

class DeepFM(tf.keras.Model):
    def __init__(self,
                 feature_columns,
                 k=10,
                 hidden_units=(200, 200, 200),
                 dnn_dropout=0.,
                 activation='relu',
                 fm_w_reg=1e-4,
                 fm_v_reg=1e-4,
                 embed_reg=1e-4):
        """
        DeepFM
        :param k: A scalar. fm's latent vector dimension
        """
        super(DeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): self.add_weight(
                name='embed_' + str(i),
                shape=(feat['feat_num'], feat['embed_dim']),
                initializer='random_uniform',
                regularizer=l2(embed_reg),
            )
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.fm = FM(k, fm_w_reg, fm_v_reg)
        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.deep_out_dense = Dense(1, activation=None)
        self.w1 = self.add_weight(name='wide_weight', shape=(1,), trainable=True)
        self.w2 = self.add_weight(name='deep_weight', shape=(1,), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,), trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs = tf.concat([inputs[feat['name']] for feat in self.dense_feature_columns], axis=1)
        sparse_embed_list = []
        for i, feat in enumerate(self.sparse_feature_columns):
            feat_value = inputs[feat['name']]
            weights = self.embed_layers['embed_{}'.format(i)]
            sparse_embed_list.append(tf.matmul(feat_value, weights))
        sparse_embed = tf.concat(sparse_embed_list, axis=1)
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1)
        wide_out = self.fm(stack)
        deep_out = self.dnn(stack)
        deep_out = self.deep_out_dense(deep_out)
        combined_out = tf.add(tf.add(self.w1 * wide_out, self.w2 * deep_out), self.bias)
        outputs = tf.nn.sigmoid(combined_out)
        return outputs