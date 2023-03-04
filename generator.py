import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM
import random



filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #works!!


characters = sorted(set(text))  
char_to_index = dict((c, i) for i, c in enumerate(characters)) #each character gets assigned a number
index_to_char = dict((i, c) for i, c in enumerate(characters))  #each number gets assigned a character

SEQ_LENGTH = 40



model = tf.keras.models.load_model("textgenerator.model") #this model rn just takes a sequence and predicts next character


# TEXT GENERATION PREDICTIONS

def sample(preds, temperature=1.0): #takes our prediciton and picks one character #higher temperature chooses characters that are more risky
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated



print("-----0.2-------")
print(generate_text(300, 0.2))
print("-----0.4-------")
print(generate_text(300, 0.4))
print("-----0.5-------")
print(generate_text(300, 0.5))
print("-----0.6-------")
print(generate_text(300, 0.6))
print("-----0.7------")
print(generate_text(300, 0.7))
print("-----0.8-------")
print(generate_text(300, 0.8))
