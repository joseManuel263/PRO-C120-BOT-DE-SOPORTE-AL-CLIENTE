# Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Palabras a ignorar/omitir mientras se crea el conjunto de datos
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Cargar la biblioteca para el modelo
import tensorflow
from data_preprocessing import get_stem_words

# Cargar el modelo
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Cargar los archivos de datos
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(stem_words, pattern_word_tags_list):

    bag=[]
    for word_tags in pattern_word_tags_list:

        pattern_words = word_tags[0] 
        bag_of_words = []
        stem_pattern_words= get_stem_words(pattern_words, ignore_words)
        for word in stem_words:            
            if word in stem_pattern_words:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # Extraer la clase desde predicted_class_label
   predicted_class = ""

   # Ahora que tenemos la etiqueta de predicción, seleccionar una respuesta aleatoria

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # Elegir una respuesta aleatoria del bot
        bot_response = ""
    
        return bot_response
    
# Nota: Las siguientes oraciones se mantienen en inglés para preservar la uniformidad del chatbot
print("Hi I am Stella, How Can I help you?")

while True:

    # Tomar input del usuario
    user_input = input('Type you message here : ')

    response = bot_response(user_input)
    print("Bot Response: ", response)
