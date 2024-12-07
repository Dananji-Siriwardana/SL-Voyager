from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the dataset and encoders
df = pd.read_csv('rp_data.csv')

# Initialize encoders
season_encoder = LabelEncoder()
type_encoder = LabelEncoder()
location_encoder = LabelEncoder()
name_encoder = LabelEncoder()

df['Season'] = season_encoder.fit_transform(df['Season'])
df['Type'] = type_encoder.fit_transform(df['Type'])
df['Location'] = location_encoder.fit_transform(df['Location'])
df['Name'] = name_encoder.fit_transform(df['Name'])

# Load the pre-trained model
model = load_model('trec_model.keras')

app = Flask(__name__)

# Functions from your notebook
def get_available_seasons():
    return season_encoder.classes_

def get_matching_types(selected_seasons):
    selected_seasons_encoded = season_encoder.transform(selected_seasons)
    filtered_data = df[df['Season'].isin(selected_seasons_encoded)]
    types = filtered_data['Type'].unique()
    return type_encoder.inverse_transform(types)

def recommend_places(selected_seasons, selected_types):
    selected_seasons_encoded = season_encoder.transform([season.lower() for season in selected_seasons])
    selected_types_encoded = type_encoder.transform([type.lower() for type in selected_types])

    filtered_data = df[(df['Season'].isin(selected_seasons_encoded)) & (df['Type'].isin(selected_types_encoded))]
    
    filtered_data['Name'] = name_encoder.inverse_transform(filtered_data['Name'])
    filtered_data['Location'] = location_encoder.inverse_transform(filtered_data['Location'])
    
    return filtered_data[['Name', 'Location']].to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    seasons = get_available_seasons()
    if request.method == 'POST':
        selected_seasons = request.form.getlist('season')
        return render_template('types.html', seasons=selected_seasons, types=get_matching_types(selected_seasons))
    return render_template('index.html', seasons=seasons)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_seasons = request.form.getlist('season')
    selected_types = request.form.getlist('type')

    print("Selected Seasons:", selected_seasons)  # Debug
    print("Selected Types:", selected_types)      # Debug

    recommendations = recommend_places(selected_seasons, selected_types)

    print("Recommendations:", recommendations)  # Debug
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
