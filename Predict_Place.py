import pandas as pd
import joblib

# Load  model
model = joblib.load('event_recommendation_model.pkl')

# Function to make recommendations
def recommend_venue(event_type, guest_count, min_budget, max_budget, special_requirements=None):
    # Prepare input for the model
    input_data = pd.DataFrame({
        'Event_Type': [event_type],
        'Guest_Count': [guest_count],
        'Min_Budget': [min_budget],
        'Max_Budget': [max_budget],
        'Special_Requirements': [special_requirements] if special_requirements else ['None']
    })

    # Make predictions
    recommended_venue = model.predict(input_data)
    return recommended_venue[0]

# Inputs
event_type_input = "Exhibition"
guest_count_input = 10
min_budget_input = 100
max_budget_input = 100
special_requirements_input = "None"

recommended_venue = recommend_venue(event_type_input, guest_count_input, min_budget_input, max_budget_input, special_requirements_input)

print(f"Recommended Venue: {recommended_venue}")
