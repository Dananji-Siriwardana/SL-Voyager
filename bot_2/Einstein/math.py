
from flask import Flask, render_template, request
from chatterbot import ChatBot

app = Flask(__name__)

# Initialize the MathBot with the MathematicalEvaluation adapter
math_bot = ChatBot(
    "Math",
    logic_adapters=["chatterbot.logic.MathematicalEvaluation"]
)

# word converter
from pint import UnitRegistry
ureg = UnitRegistry()

def convert_units(query):
    try:
        if "convert" in query:
            words = query.split()
            value = float(words[1])
            from_unit = words[2]
            to_unit = words[-1]
            result = value * ureg(from_unit).to(to_unit)
            return f"{value} {from_unit} is equal to {result.magnitude:.2f} {to_unit}"
    except Exception:
        return "Sorry, I couldn't process the conversion."


@app.route("/")
def main():
    return render_template("index.html")  # Main chatbot interface

@app.route("/mathbot")
def mathbot():
    return render_template("mathbot.html")  # MathBot interface

@app.route("/get")
def get_response():
    userText = request.args.get('userMessage')  # Get user input
    try:
        # Check for conversion queries
        if "convert" in userText.lower():
            return convert_units(userText)
        
        # Otherwise, default to math evaluation using MathBot
        return str(math_bot.get_response(userText))
    except Exception as e:
        return f"Sorry, an error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
