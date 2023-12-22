import numpy as np
import tensorflow as tf

from strangefish.models.model_training_utils import masked_squared_error

keras = tf.keras


# from https://keras.io/examples/nlp/text_classification_with_transformer/
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def uncertainty_transformer_1(num_heads=2, ff_dim=64, dropout=0.1, weights_path=None):

    input_dim = (None, 8, 8, 37)

    # Input data type
    dtype = 'float32'

    # ---- Network model ----
    input_data = keras.layers.Input(name='input', shape=input_dim, dtype=dtype)

    x = keras.layers.Masking(mask_value=-1.0)(input_data)

    input_len = np.product(input_dim[1:])

    x = keras.layers.Reshape((-1, input_len), name='reshape_1')(x)

    x = TransformerBlock(input_len, num_heads, ff_dim, dropout)(x)

    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(units=64, activation='relu', name='fc')(x)
    x = keras.layers.Dropout(dropout, name='dropout_3')(x)

    y_pred = keras.layers.Dense(1, name='output', activation='sigmoid')(x)

    model = keras.models.Model(inputs=input_data, outputs=y_pred)

    model.compile(loss=masked_squared_error, optimizer='adam')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def uncertainty_transformer_vis(num_heads=2, ff_dim=64, dropout=0.1, weights_path=None):

    input_dim = (None, 8, 8, 37)

    # Input data type
    dtype = 'float32'

    # ---- Network model ----
    input_data = keras.layers.Input(name='input', shape=input_dim, dtype=dtype)

    x = keras.layers.Masking(mask_value=-1.0, name="masking")(input_data)

    input_len = np.product(input_dim[1:])

    x = keras.layers.Reshape((-1, input_len), name='reshape')(x)

    x = TransformerBlock(input_len, num_heads, ff_dim, dropout)(x)

    x = keras.layers.Dropout(dropout, name="dropout_1")(x)

    x = keras.layers.Dense(units=64, activation='relu', name='dense')(x)
    x = keras.layers.Dropout(dropout, name='dropout_2')(x)

    y_pred = keras.layers.Dense(1, name='output', activation='sigmoid')(x)

    model = keras.models.Model(inputs=input_data, outputs=y_pred)

    model.compile(loss=masked_squared_error, optimizer='adam')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    model = uncertainty_transformer_1()
