import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from ctr_model.layers.layers import DNN, Linear


class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units=(200, 200), activation='relu', dnn_dropout=0., embed_reg=1e-4):
        super(WideDeep, self).__init__()
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
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.deep_out_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        dense_inputs = tf.concat([inputs[feat['name']] for feat in self.dense_feature_columns], axis=1)
        sparse_embed_list = []
        for i, feat in enumerate(self.sparse_feature_columns):
            feat_value = inputs[feat['name']]
            weights = self.embed_layers['embed_{}'.format(i)]
            sparse_embed_list.append(tf.matmul(feat_value, weights))
        sparse_embed = tf.concat(sparse_embed_list, axis=1)
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1)
        wide_out = self.linear(dense_inputs)
        deep_out = self.dnn_network(stack)
        deep_out = self.deep_out_dense(deep_out)
        outputs = tf.nn.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

