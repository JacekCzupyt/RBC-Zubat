import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, Activation, Dropout, LSTM, Dense, Reshape, Masking
from tensorflow.python.keras.models import Model


def uncertainty_model_1(weights_path):
    def masked_squared_error(y_true, y_pred):
        mask = tf.where(tf.not_equal(y_true, -1.0))
        return tf.reduce_mean(tf.square(tf.gather(y_true, mask) - tf.gather(y_pred, mask)))

    input_dim = (8, 8, 37)

    dropout = 0

    # Input data type
    dtype = 'float32'

    # ---- Network model ----
    input_data = Input(name='input', shape=input_dim, dtype=dtype)

    x = Masking(mask_value=-1)(input_data)

    x = Conv2D(filters=128, kernel_size=3, name='conv_1')(x)
    x = BatchNormalization(name='norm_1')(x)
    x = Activation('relu', name='activation_1')(x)
    x = Dropout(dropout, name='dropout_1')(x)

    x = Conv2D(filters=256, kernel_size=3, name='conv_2')(x)
    x = BatchNormalization(name='norm_2')(x)
    x = Activation('relu', name='activation_2')(x)
    x = Dropout(dropout, name='dropout_2')(x)

    x = Reshape((-1, 4*4*256), name='reshape_1')(x)

    x = LSTM(128, activation='relu', return_sequences=True,
             dropout=dropout, name='lstm_1')(x)
    x = LSTM(128, activation='relu', return_sequences=True,
             dropout=dropout, name='lstm_2')(x)

    x = Dense(units=64, activation='relu', name='fc')(x)
    x = Dropout(dropout, name='dropout_3')(x)

    y_pred = Dense(1, name='output', activation='sigmoid')(x)

    model = Model(inputs=input_data, outputs=y_pred)

    model.compile(loss=masked_squared_error, optimizer='adam')

    model.load_weights(weights_path)

    return model



