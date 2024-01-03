import os
import sys

import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split

from strangefish.models.model_training_utils import GameHistorySequence
from strangefish.models.uncertainty_transformer import uncertainty_transformer_1
from strangefish.models.uncertainty_lstm import uncertainty_lstm_1


if __name__ == '__main__':
    os.chdir('..')

    match sys.argv[1]:
        case 'lstm':
            model = uncertainty_lstm_1()
        case 'trans':
            model = uncertainty_transformer_1()
        case _:
            raise Exception(f"Unknown model type: {sys.argv[1]}")

    print(model.summary())
    model_name = sys.argv[2]

    epochs = int(sys.argv[3])
    init_epoch = 0

    if len(sys.argv) > 4:
        model.load_weights(f'uncertainty_model/{sys.argv[4]}/weights')
        if os.path.isfile(f'uncertainty_model/{sys.argv[4]}/model_history_log.csv'):
            init_epoch = len(pd.read_csv(f'uncertainty_model/{sys.argv[4]}/model_history_log.csv'))
    data_path = 'game_logs/historical_games_extended'

    if os.path.isfile('uncertainty_model/train_data.csv') and os.path.isfile('uncertainty_model/test_data.csv'):
        train_data = list(pd.read_csv('uncertainty_model/train_data.csv')['files'])
        test_data = list(pd.read_csv('uncertainty_model/test_data.csv')['files'])
    else:
        files = os.listdir(data_path)
        files.sort()

        train_data, test_data = train_test_split(files, test_size=0.2, random_state=42)

        pd.DataFrame({'files': train_data}).to_csv('uncertainty_model/train_data.csv')
        pd.DataFrame({'files': test_data}).to_csv('uncertainty_model/test_data.csv')

    training_sequence = GameHistorySequence(train_data, data_path, 16)
    test_sequence = GameHistorySequence(test_data, data_path, 16, shuffle=False)

    checkpoint = ModelCheckpoint(f'uncertainty_model/{model_name}', 'val_loss', verbose=1, mode='max')
    csv_logger = CSVLogger(f"uncertainty_model/{model_name}/model_history_log.csv", append=True)

    model.fit(
        training_sequence,
        validation_data=test_sequence,
        epochs=epochs+init_epoch,
        initial_epoch=init_epoch,
        callbacks=[checkpoint, csv_logger]
    )

    try:
        model.save(f'uncertainty_model/{model_name}')
    except Exception as e:
        print(e)

    model.save_weights(f'uncertainty_model/{model_name}/weights')