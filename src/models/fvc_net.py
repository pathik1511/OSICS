import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

def get_residual_block(x, units, dropout_rate):
    shortcut = x
    x = L.Dense(units, activation='swish')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout_rate)(x)
    x = L.Dense(units, activation='swish')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout_rate)(x)
    x = L.Add()([shortcut, x])
    return x

def quantile_sum(x):
    return x[0] + tf.cumsum(x[1], axis=1)

def build_model(input_dim, config):
    inputs = L.Input(shape=(input_dim,), name="Patient_Features")

    # Initial projection
    x = L.Dense(config['hidden_units'], activation='swish')(inputs)
    x = L.BatchNormalization()(x)

    # Residual blocks
    for _ in range(config['num_res_blocks']):
        x = get_residual_block(x, config['hidden_units'], config['dropout_rate'])

    # Quantile regression heads
    p1 = L.Dense(3, activation='linear', name="p1")(x)
    p2 = L.Dense(3, activation='relu', name="p2")(x)

    preds = L.Lambda(quantile_sum, name="preds")([p1, p2])

    model = M.Model(inputs=inputs, outputs=preds, name="EnterpriseFVCNet")
    return model
