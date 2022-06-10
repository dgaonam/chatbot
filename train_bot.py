from base64 import encode
import nltk
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

import json
import pickle

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import SGD

import warnings
import logging, platform, os

import tensorflow

words = []
classes = []
documents = []
ignore_words = ['?', '!',',','.','te','la','es']
data_file = open('intents.json',encoding='utf8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.lower()
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag'].lower()))
        if intent['tag'].lower() not in classes:
            classes.append(intent['tag'].lower())
           
words = [w.lower() for w in words if w not in ignore_words]
pickle.dump(words, open('data/words.pkl','wb'))
pickle.dump(classes, open('data/classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]

    pattern_words = [word.lower() for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training)
# creazione dei set di train e di test: X - patterns, Y - intents
print("list X: " + str(list(training[:,0])))
print("list X: " + str(list(training[:,1])))
x_val = list(training[0:])
y_val = list(training[1:])
train_x = list(training[:,0])
train_y = list(training[:,1])

# creazione del modello
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer =  tensorflow.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=10, verbose=0)#validation_data=(x_val,y_val)
print("Evaluate model on test data")
results = model.evaluate(np.array(train_x), np.array(train_y), batch_size=128)
print("test loss, test acc:", results)
model.save('data/chatbot_model.h5', hist)

print("model created")