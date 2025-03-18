'''
from chatterbot import ChatBot

bot = ChatBot("Math" , logic_adapters = ["chatterbot.logic.MathematicalEvaluation"])

print("--------Einstein, the MathBot--------")

while True:
    user_text = input("type the equation : ")
    print("Einstein : " + str(bot.get_response(user_text)))
    '''

from chatterbot import ChatBot

# Initialize the ChatBot with the MathematicalEvaluation logic adapter
bot = ChatBot(
    "Math",
    logic_adapters=["chatterbot.logic.MathematicalEvaluation"]
)

print("--------Einstein, the MathBot--------")
print("[Use spaces when typing numbers as well]")


while True:
    try:
        user_text = input("Type the equation: ")
        
        # Check if the user wants to exit
        if user_text.lower() in ["exit", "quit", "bye"]:
            print("Einstein: lim x→∞​ f(x) = Farewell!")
            break
        
        # Get and print the response
        response = bot.get_response(user_text)
        print("Einstein: " + str(response))
    except Exception as e:
        print(f"Einstein: Sorry, I couldn't understand that. Error: {e}")
