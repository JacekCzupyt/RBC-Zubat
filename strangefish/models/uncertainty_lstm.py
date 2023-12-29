import tensorflow as tf

from strangefish.models.model_training_utils import masked_squared_error

# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv2D, Activation, Dropout, LSTM, Dense, Reshape, Masking
# from tensorflow.python.keras.models import Model

keras = tf.keras


def uncertainty_lstm_1(weights_path=None):
    input_dim = (None, 8, 8, 37)

    dropout = 0

    # Input data type
    dtype = 'float32'

    # ---- Network model ----
    input_data = keras.Input(name='input', shape=input_dim, dtype=dtype)

    x = keras.layers.Masking(mask_value=-1.0)(input_data)

    x = keras.layers.Conv2D(filters=128, kernel_size=3, name='conv_1')(x)
    x = keras.layers.BatchNormalization(name='norm_1')(x)
    x = keras.layers.Activation('relu', name='activation_1')(x)
    x = keras.layers.Dropout(dropout, name='dropout_1')(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=3, name='conv_2')(x)
    x = keras.layers.BatchNormalization(name='norm_2')(x)
    x = keras.layers.Activation('relu', name='activation_2')(x)
    x = keras.layers.Dropout(dropout, name='dropout_2')(x)

    x = keras.layers.Reshape((-1, 4 * 4 * 256), name='reshape_1')(x)

    x = keras.layers.LSTM(128, activation='relu', return_sequences=True,
             dropout=dropout, name='lstm_1')(x)
    x = keras.layers.LSTM(128, activation='relu', return_sequences=True,
             dropout=dropout, name='lstm_2')(x)

    x = keras.layers.Dense(units=64, activation='relu', name='fc')(x)
    x = keras.layers.Dropout(dropout, name='dropout_3')(x)

    y_pred = keras.layers.Dense(1, name='output', activation='sigmoid')(x)

    model = keras.Model(inputs=input_data, outputs=y_pred)

    model.compile(loss=masked_squared_error, optimizer='adam')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model
