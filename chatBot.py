
import speech_recognition as sr
import pyttsx3
import  nltk
import os
import torch
import tensorflow
from keras.models import load_model
import json,pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#import spacy
#import stanza
#from spacy_stanza import StanzaLanguage

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

#import AutoTokenizer, AutoModelForCausalLM, Pipeline, Conversation
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device = torch.device("cuda")

engine = pyttsx3.init()

#RATE DEL ASISTENTE
rate = engine.getProperty('rate')
engine.setProperty('rate', 100)

#VOZ DEL ASISTENTE
voice_engine = engine.getProperty('voices')
engine.setProperty('voice', voice_engine[1].id)

class chat_Bot():
    def __init__(self,name):
        print("--iniciando",name,"---")
        self.name = name
    
    def speech_to_text(self):
#        recognizer = sr.Recognizer()
#        with sr.Microphone() as micro:
            print("Escuchando...")
#            audio = recognizer.listen(micro)
#            try:
#                self.text = recognizer.recognize_google(audio,language='es-MX',show_all=False)
#                print("Escucho: " + self.text)
#            except BaseException as err:
#                print(f"Unexpected {err=}, {type(err)=}")
#                print("no se capto ningun audio")
#                self.text = None
    
    def wake_up(self,text):
        return True if self.name in text.lower() else False
    
    @staticmethod
    def text_to_speech(self,text):
        print("IA[ " + self.name.upper() +" ]" + text) 
        self.talk(text)

    @staticmethod        
    def talk(text):
        engine.say(text)
        engine.runAndWait()
        print("\n") 

    # preprocessamento input utente
    def clean_up_sentence(self,sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.word.lower() for self.word in sentence_words]
        return sentence_words

    # creazione bag of words
    def bow(self,sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(self.words)
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        
        return(np.array(bag))

    def calcola_pred(self,sentence):
        p = self.bow(sentence,show_details=True)
    
        res = self.model.predict(np.array([p]))[0]
        results = []
        ERROR_THRESHOLD = 0.10
    
        for i,r in enumerate(res): 
            if r>ERROR_THRESHOLD: 
                print("r" + str(r))
                results.append([i,r,ERROR_THRESHOLD])

        results.sort(key=lambda x: x[1], reverse=True)
        print("Result: " + str(results))
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1]), "ERROR_THRESHOLD": str(r[2]) })
   
        return return_list

    def buscar(self,ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']

        for i in list_of_intents:
            if(i['tag'].lower()== tag.lower()):
                result = random.choice(i['responses'])
                break
        return result   

    @staticmethod 
    def respond(self,text):
        ints = self.calcola_pred(text)
        res = self.buscar(ints)
        print(str(ints))
        print("IA [" + self.name.upper() +"]" + str(res))
        self.talk(res)

if __name__=="__main__":
    ai = chat_Bot(name="maya")
    ai.model = load_model("data/chatbot_model.h5")
    ai.words = pickle.load(open('data/words.pkl','rb'))
    ai.classes = pickle.load(open('data/classes.pkl','rb'))
    ai.intents = json.loads(open('intents.json',encoding='utf8').read())
    ai.text_to_speech(ai,"Bienvenido")
    while True:
        #ai.speech_to_text()
        
        ai.text = str(input(""))
        if ai.text is not None:
            try:
                if ai.wake_up(ai.text) is True:
                    res="Hola soy " + ai.name +", la IA, ¿ Que puedo hacer por ti?"
                    ai.text_to_speech(ai,res)
                elif any(i == ai.text.lower() for i in ["salir","bye","adios","no"]):
                    ai.text_to_speech(ai,"Adios")
                    break  
                elif ai.text not in "":
                    ai.respond(ai,ai.text)   
                else: 
                    ai.text_to_speech(ai,"sin info")       
            except (RuntimeError, TypeError, NameError):
                print("Error: ->" + str(RuntimeError) + str(TypeError) + str(NameError))                  