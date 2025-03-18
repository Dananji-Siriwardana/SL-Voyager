import nltk
#nltk.download('punkt_tab')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

import requests

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer , ChatterBotCorpusTrainer
from flask import Flask , render_template , request

app = Flask(__name__)

bot = ChatBot("Emotional Assistant" , read_only = False , 
    logic_adapters = [
        {
            "import_path" : "chatterbot.logic.BestMatch",
            "default_response" : "Sorry. I don't know.",
            "maximum_similarity_threshold" : 0.9}
        ])


list_to_train_01 = [
    #test
    "what can you do?",
    "I can assist with various tasks, answer questions, and chat with you!",
    "how can you help me?",
    "I can provide information, offer suggestions, and more.",
    "where are you from?",
    "I'm a virtual assistant created to help you.",
    "what's your favorite color?",
    "I like all colors equally!",
    "can you tell me a joke?",
    "Sure! Why did the computer go to the doctor? It caught a virus!"
]


list_to_train_02 = [
    "hi",
    "hi there",
    "what's your name?",
    "I'm a ChatBot!",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you?",
    "I'm doing well, thank you!",
    "nice to meet you",
    "pleased to meet you"
]

list_to_train_03 = [
    "hi",
    "sup",
    "hotels?",
    "visit : https://google.com"
]

list_trainer = ListTrainer(bot)

list_trainer.train(list_to_train_01)
list_trainer.train(list_to_train_02)
list_trainer.train(list_to_train_03)




'''
THE_DUMB_BOT

trainer = ChatterBotCorpusTrainer(bot)

trainer.train("chatterbot.corpus.english")

'''






@app.route( "/" )

def main():
    return render_template( "index.html" )




@app.route("/get")
def get_chatbot_response():
    userText = request.args.get('userMessage')

    rawData = requests.get("https://api.open-meteo.com/v1/forecast?latitude=6.9&longitude=79.8&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    result = rawData.json()

    return result #str(bot.get_response(userText))



if __name__ == "__main__":
    app.run ( debug = True )




while True :
    user_response = input("User : ")
    print("Velora : " + str(bot.get_response(user_response)))
