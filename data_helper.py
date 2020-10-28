import numpy as np
import json


class FeatureHelper:
    def __init__(self, feature_config_file):
        self.dense_features, self.sparse_features = self.parse_feature_config(feature_config_file)

    def parse_feature_config(self, feature_config_file):
        file_obj = open(feature_config_file, "r")
        feature_obj = json.load(file_obj)
        dense_features = dict(feature_obj['USER']['vector_features'], **feature_obj['ITEM']['vector_features'])
        sparse_features = dict(feature_obj['USER']['embedding_features'], **feature_obj['ITEM']['embedding_features'])
        return dense_features, sparse_features

    def sparseFeature(self, feat, feat_num, embed_dim=4):
        return {'name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

    def denseFeature(self, feat, feat_num):
        return {'name': feat, 'feat_num': feat_num}

    def format_input_features(self, df):
        dense_feature_cols = [self.denseFeature(feat, self.dense_features[feat]['feature_length']) for feat in self.dense_features]
        sparse_feature_cols = [self.sparseFeature(feat, self.sparse_features[feat]['feature_length'], self.sparse_features[feat]['embed_dim']) for feat in self.sparse_features]
        feature_columns = [dense_feature_cols, sparse_feature_cols]
        feature_names = list(self.dense_features.keys()) + list(self.sparse_features.keys())
        features = {feat: np.array(df[feat].tolist()) for feat in feature_names}
        label = np.array(df['label'].tolist())
        return feature_columns, features, label


if __name__ == '__main__':
    feature_helper = FeatureHelper("./feature_config.json")


