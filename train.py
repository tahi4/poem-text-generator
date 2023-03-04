import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM


# ACCESSING FILE
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() 

characters = sorted(set(text))  # ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_index = dict((c, i) for i, c in enumerate(characters)) #each character gets assigned a number
index_to_char = dict((i, c) for i, c in enumerate(characters))  #each number gets assigned a character

SEQ_LENGTH = 40 #each sentence has 40 characters
STEP_SIZE = 3


sentences = [] #how are yo
next_char = [] #u

for i in range (0, len(text) - SEQ_LENGTH, STEP_SIZE):

    sentences.append(text[i:i+SEQ_LENGTH]) #1 to 39 #first citizen: before we proceed any fur
    next_char.append(text[i+SEQ_LENGTH]) #40 #t


x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_) #how many sentnces we have * the length of one sentence * amount of possible characters
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_) #which character would be next in a sentence

for i, sentence in enumerate(sentences): #take sentence array and assign an index like 0th sentence, 1st etc etc
    for t, character in enumerate(sentence): #for each of those characters in a sentence, t is the position of the sentence character 
        x[i, t, char_to_index[character]] = 1 #sentence no. i, at position t, at character number this position is set to 1
    y[i, char_to_index[next_char[i]]] = 1 #y-target data# this sentence, the next character is this one

# NEURAL NETWORK

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters)))) #memory of our model #128 neurons, input shape of 40x38
model.add(Dense(len(characters))) #no. of neurons = length of characters
model.add(Activation('softmax')) #softmax gives probability from 0 to 1

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4) #batch size = how many example will we put in at once #epoch = the np. of times itll see this same data


model.save("textgenerator.model")









# on line 26 i previously made the ERROR in the for loop of giving next_char.append(text[SEQ_LENGTH]) instead of next_char.append(text[i+SEQ_LENGTH])
# next_char.append(text[SEQ_LENGTH]) means will only train the model on the FIRST sentence's next character
# next_char.append(text[i+SEQ_LENGTH]) means it trains the model on the next character for EVERY sentence
# which caused the error of only generating t's, instead of actual text