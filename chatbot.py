import random
import json
import pickle
import numpy as np
import nltk
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

mobilePrices = pd.read_csv("Mobiles.csv")

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

brand = None
pmodel = None
color = None
memory = None
storage = None
sp = None
op = None


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def SellAndOrg(brand, pmodel, color, memory, storage):
    loc = 0
    column_to_search = 'Brand'
    sp = 0
    op = 0

    for i in range (len(mobilePrices)):
        
        if mobilePrices.Brand[i] == brand and mobilePrices.Model[i] == pmodel and mobilePrices.Color[i] == color and mobilePrices.Memory == memory and mobilePrices.Storage == storage:
            loc = i
            break
        
    sp = mobilePrices['Selling Price'][i].astype(int)
    op = mobilePrices['Original Price'][i].astype(int)

    return sp, op

def generate_random_series(low, high, num_values):
    
    random_values = np.random.randint(low, high, size=num_values)
    return random_values

sp, op = SellAndOrg(brand, pmodel, color, memory, storage)

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break

    if "___" in result:
        high = generate_random_series(sp, op, 1)
        op = high
        result = result.replace("___", high)
    
    return result





print("GO! Bot is running!")

while True:
    while True:
        if brand == None and pmodel == None and color == None and memory == None :
            print("""
                The Inputs must be in the format:
                Brand: OPPO
                Model: A53
                Color: Moonlight Black
                Memory: 4 GB
                Storage: 64 GB

                """)
            print("Now fill in the below values to continue:")
            brand = input("Brand: ")
            pmodel = input("model: ")
            color = input("Color: ")
            memory = input("Memory: ")
            storage = input("Storage: ")

            print("Thank You")
            print("An AI Agent will be assigned to you soon......")
        else:
            break



    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print ("Agent: "+res)