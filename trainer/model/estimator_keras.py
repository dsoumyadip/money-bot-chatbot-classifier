import os
import json
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from trainer.utils.utils import Utils

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = len(Utils.CHAR_DICT.keys())


def train_and_evaluate(data: pd.DataFrame, args: Dict):
    x = pad_sequences(data['CLIENT'], maxlen=args['sequence_length'])
    print('Shape of data tensor:', x.shape)

    y = pd.get_dummies(data['ACTIVITY']).values
    print('Shape of label tensor:', y.shape)

    activity_dict = dict(zip(list(pd.get_dummies(data['ACTIVITY']).columns), range(len(list(pd.get_dummies(data['ACTIVITY']).columns)))))

    with open('resources/activities2dummies.json', 'w') as fp:
        json.dump(activity_dict, fp)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args['train_test_ratio'], random_state=42)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, args['embedding_size'], input_length=x.shape[1]))
    model.add(SpatialDropout1D(args['keep_prob']))
    model.add(LSTM(args['sequence_length'], dropout=args['keep_prob'], recurrent_dropout=args['keep_prob']))
    model.add(Dense(len(data['ACTIVITY'].unique()), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=args['train_steps'], batch_size=args['batch_size'],
                        validation_split=args['validation_split'],
                    )

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(os.path.join(args['base_dir'], args['output_dir'], "loss"))

    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.savefig(os.path.join(args['base_dir'], args['output_dir'], 'accuracy'))

    model.save(os.path.join(args['base_dir'], args['output_dir'], 'lstm_model.h5'))  # creates a HDF5 file 'my_model.h5'

    print("Model training completed.")

