import nltk

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer , ChatterBotCorpusTrainer
from flask import Flask , render_template

app = Flask(__name__)

bot = ChatBot("Emotional Assistant" , read_only = False , 
    logic_adapters = [
        {
            "import_path" : "chatterbot.logic.BestMatch",
            "default_response" : "Sorry. I don't know.",
            "maximum_similarity_threshold" : 0.9}
        ])

trainer = ChatterBotCorpusTrainer(bot)

trainer.train("chatterbot.corpus.english")





while True :
    user_response = input("User : ")
    print("Velora : " + str(bot.get_response(user_response)))


