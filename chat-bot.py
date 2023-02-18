import functools
from pyexpat import model
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, Conversation
import json

engine = pyttsx3.init()

#RATE DEL ASISTENTE
rate = engine.getProperty('rate')
engine.setProperty('rate', 150)

#VOZ DEL ASISTENTE
voice_engine = engine.getProperty('voices')
print("Voces: " + str(voice_engine))
engine.setProperty('voice', voice_engine[0].id)

json_filename = 'config.json'

class chat_Bot():
    @functools.lru_cache(maxsize = None) 
    def __init__(self,name):
        print("--iniciando",name,"---")
        self.name = name
        #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        #self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium",from_tf=True, config=json_filename)
        #assert self.model.config.output_attentions == True
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as micro:
            print("Escuchando...")
            audio = recognizer.listen(micro)
            try:
                self.text = recognizer.recognize_google(audio,language='es-MX',show_all=False)
                print("Escucho: " + self.text)
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
    def wake_up(self,text):
        return True if self.name in text.lower() else False
    @staticmethod
    def text_to_speech(self,text):
        print("ai->" + text) 
        self.talk(text)
    @staticmethod        
    def talk(text):
        engine.say(text)
        engine.runAndWait() 
    @staticmethod 
    def respond(self,text):
         #, model = self.model, tokenizer = self.tokenizer
         Pipeline('conversational', AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium",from_tf=True, config=json_filename), AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium"))
         res = Conversation(text)
         print("R " + str(res))
         self.talk(res)

        

if __name__=="__main__":
    ai = chat_Bot(name="dani")
    while True:
        ai.speech_to_text()
        if ai.wake_up(ai.text) is True:
            res="Hola soy Dani, la IA, ¿ Que puedo hacer por ti?"
            ai.text_to_speech(ai,res)
        elif any(i in ai.text for i in ["salir","Bye","Adios","No"]):
            ai.text_to_speech(ai,"Adios")
            break  
        else:
            ai.respond(ai,ai.text)  