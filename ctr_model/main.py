import sys
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from sklearn.externals import joblib
from ctr_model.data_helper import FeatureHelper


def get_model(model_name):
    if model_name == 'LR':
        from ctr_model.models.LR import LR
        model = LR(feature_columns=feature_columns)
    elif model_name == 'FM':
        from ctr_model.models.FM import FM
        model = FM(feature_columns=feature_columns, k=k)
    elif model_name == 'DeepFM':
        from ctr_model.models.DeepFM import DeepFM
        model = DeepFM(feature_columns=feature_columns, k=k)
    elif model_name == 'WideDeep':
        from ctr_model.models.WideDeep import WideDeep
        model = WideDeep(feature_columns=feature_columns)
    else:
        raise ValueError("Not supported model ", model_name)
    return model


if __name__ == '__main__':
    df = joblib.load("./data.pkl")
    feature_helper = FeatureHelper("./feature_config.json")
    feature_columns, features, label = feature_helper.format_input_features(df)
    k = 10
    learning_rate = 0.001
    batch_size = 64
    epochs = 5

    args = sys.argv
    model_name = args[1]
    model = get_model(model_name)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])
    model.fit(features, label, epochs=epochs, batch_size=batch_size)
    model.save("./test_model")