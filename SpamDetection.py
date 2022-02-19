import random
import json

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.config import Config

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import numpy as np

import warnings
warnings.filterwarnings("ignore")


Window.size = (500, 300)
Window.clearcolor = (27/255, 36/255, 52/255, 1)
Config.set('graphics', 'resizable', '0')

VOCAB_SIZE = 2000
EMBEDDING_DIM = 128
MAX_LENGTH = 100
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"

with open('Model/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

model = tf.keras.models.load_model('Model/Model.h5')

data = pd.read_csv('spam.csv', encoding='latin-1')
messages = data['v2'].to_list()

stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(sentence):
    word_tokens = nltk.tokenize.word_tokenize(sentence)
    filtered_sentence = [w.lower() for w in word_tokens if w.isalpha()]
    filtered_sentence = [w for w in filtered_sentence if w not in stopwords]
    return ' '.join(filtered_sentence)


def prep_text(sentence):
    pred_data = tokenizer.texts_to_sequences([sentence])
    pred_data = pad_sequences(pred_data,
                              maxlen=MAX_LENGTH,
                              padding=PADDING_TYPE,
                              truncating=TRUNC_TYPE)
    return pred_data


class Boxes(Widget):

    def predict(self):
        text = self.ids.input.text
        text = remove_stopwords(text)

        prob = model.predict(prep_text(text))
        prob = np.squeeze(prob)

        if prob > 0.8:
            self.ids.pred_text.text = 'It is a Spam'
        else:
            self.ids.pred_text.text = 'It is not a Spam'

    def generate_random(self):
        self.ids.input.text = random.choice(messages)

    def process(self):
        pass


class SpamDetection(App):

    def build(self):
        self.icon = "favicon.png"
        layout = Boxes()
        return layout
    pass


if __name__ == "__main__":
    SpamDetection().run()