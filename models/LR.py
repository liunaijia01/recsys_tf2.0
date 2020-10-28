import tensorflow as tf
from tensorflow.keras.regularizers import l2
from ctr_model.layers.layers import Linear


class LR(tf.keras.Model):
    def __init__(self, feature_columns, embed_reg=1e-4):
        super(LR, self).__init__()
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
        self.linear = Linear()

    def call(self, inputs, **kwargs):
        dense_inputs = tf.concat([inputs[feat['name']] for feat in self.dense_feature_columns], axis=1)
        sparse_embed_list = []
        for i, feat in enumerate(self.sparse_feature_columns):
            feat_value = inputs[feat['name']]
            weights = self.embed_layers['embed_{}'.format(i)]
            sparse_embed_list.append(tf.matmul(feat_value, weights))
        sparse_embed = tf.concat(sparse_embed_list, axis=1)
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1)
        linear_out = self.linear(stack)
        output = tf.nn.sigmoid(linear_out)
        return output

