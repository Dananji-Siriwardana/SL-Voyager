from flask import Flask, render_template, request
import pandas as pd
import joblib
import requests  

# Load the trained model and dataset
model = joblib.load('event_recommendation_model.pkl')
dataset = pd.read_csv('Dataset.csv')   

 
app = Flask(__name__)

# get coordinates using Google Maps API
def get_coordinates(place):
    API_KEY = 'AIzaSyA2MXdyzbpEbtQZxVLdBFQUg9qO_3ASknI'   
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={place}&key={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return {'lat': location['lat'], 'lng': location['lng']}
        elif data['status'] == 'ZERO_RESULTS':
            print(f'Error finding coordinates for {place}: ZERO_RESULTS')
            return None
        else:
            print(f'Error finding coordinates for {place}: {data["status"]}')
            return None
    else:
        print(f'Error fetching coordinates for {place}: {response.status_code}')
        return None

#  venue recommendations
def recommend_venue(event_type, guest_count, min_budget, max_budget, special_requirements=None):
    special_requirements = special_requirements if special_requirements else "None"

    input_data = pd.DataFrame({
        'Event_Type': [event_type],
        'Guest_Count': [guest_count],
        'Min_Budget': [min_budget],
        'Max_Budget': [max_budget],
        'Special_Requirements': [special_requirements]
    })

    probabilities = model.predict_proba(input_data)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_venues = model.named_steps['classifier'].classes_[top_3_indices]
    top_3_probabilities = probabilities[top_3_indices]

    recommendations = []
    for venue, prob in zip(top_3_venues, top_3_probabilities):
        venue_details = dataset[dataset['Venue_Name'] == venue].iloc[0].to_dict()
        recommendations.append({
            'name': venue,
            'probability': prob,
            'details': venue_details
        })
    
    return recommendations

#   recommendation route
@app.route("/recommended", methods=["GET", "POST"])
def index():
    recommendations = []
    map_locations = []

    if request.method == "POST":
        event_type = request.form.get("event_type")
        guest_count = int(request.form.get("guest_count"))
        min_budget = int(request.form.get("min_budget"))
        max_budget = int(request.form.get("max_budget"))
        special_requirements = request.form.get("special_requirements")
        recommendations = recommend_venue(event_type, guest_count, min_budget, max_budget, special_requirements)

        for venue in recommendations:
            address = venue['details'].get('Location', '')
            coordinates = get_coordinates(address)
            if coordinates:
                map_locations.append({
                    'name': venue['name'],
                    'address': address,
                    'type': venue['details'].get('Venue_Type', ''),
                    'rating': venue['details'].get('User_Rating', ''),
                    'probability': venue['probability'],
                    'lat': coordinates['lat'],
                    'lng': coordinates['lng']
                })

    return render_template("index.html",
                           recommendations=recommendations,
                           map_locations=map_locations,
                           form_data=request.form if request.method == "POST" else None)

 
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
