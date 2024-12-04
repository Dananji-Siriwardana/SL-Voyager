from chatterbot import ChatBot



bot = ChatBot("units" , logic_adapters = ['chatterbot.logic.UnitConversion'])

print("----------------Unit Converter----------------")
print("[format : <AMOUNT><space><UNIT><space>TO<space><UNIT>]")

while True :
    user_text = input ("Ask : ")
    chatbot_response = str(bot.get_response(user_text))
    print(chatbot_response)