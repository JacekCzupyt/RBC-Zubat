import os

import chess
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.utils import data_utils

from game_log_reader.game_history_extended import GameHistoryExtended
from strangefish.utilities.chess_model_embedding import map_game_from_history


class GameHistorySequence(tf.keras.utils.Sequence):
    def __init__(self, files, data_path, batch_size=1, shuffle=True):
        self.data_path = data_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.files = files
        if self.shuffle:
            np.random.shuffle(self.files)

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        files = self.files[idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.files))]

        histories = [GameHistoryExtended.from_file(os.path.join(self.data_path, file_name)) for file_name in files]

        data = [map_game_from_history(game, color) for game in histories for color in chess.COLORS]
        data = [e for e in data if e is not None]

        X, y = [e[0] for e in data], [e[1] for e in data]

        x_len = max(len(x) for x in X)
        X = np.array([np.concatenate((x, -1 * np.ones((x_len - x.shape[0], *x.shape[1:])))) for x in X])
        y = np.array([np.concatenate((e, -1 * np.ones((x_len - len(e))))) for e in y])

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)


def masked_squared_error(y_true, y_pred):
    # print(y_true.numpy().shape, y_pred.numpy().shape)
    mask = tf.where(tf.not_equal(y_true, -1.0))
    return tf.reduce_mean(tf.square(tf.gather_nd(y_true, mask) - tf.gather_nd(y_pred, mask)))
