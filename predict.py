import json
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from trainer.utils.utils import Utils

with open('resources/activities2dummies.json', 'r') as fp:
    activities_dict = json.load(fp)

reverse_dict = {v: k for k,v in activities_dict.items()}

model = load_model('/home/soumyadip/Desktop/PycharmProjects/intent-activity-classifier/output/lstm_model.h5')
while True:
    user_input = input("You:  ")
    user_input = Utils.clean_sentence(user_input)
    user_input = list(user_input)
    user_input = [Utils.CHAR_DICT[i] for i in user_input]
    y_prob = model.predict(pad_sequences(np.array([user_input]), maxlen=30))
    y_class = y_prob.argmax(axis=-1)
    print("Bot:  " + reverse_dict[y_class[0]])