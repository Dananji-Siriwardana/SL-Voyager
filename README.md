# **SL Voyager: An AI-Integrated Tourism Management System**

---

## **Overview**
SL Voyager is an innovative AI-driven platform designed to enhance the travel experience in Sri Lanka. By integrating cutting-edge technology, the system provides personalized recommendations, emotional support, and cultural insights, making it an all-in-one solution for tourists and tourism operatives.

---

## **Features**
1. **Attractions Suggestion System**  
   - Personalized recommendations for must-visit attractions.  
   - Includes cultural, scenic, and adventure spots.  

2. **Hotel Recommendation System**  
   - Tailored hotel suggestions based on preferences and budgets.  
   - Includes filters like family-friendly, luxury, and eco-friendly options.  

3. **Tourism Forecasting System**  
   - Predicts peak and off-season trends for optimized planning.  
   - Provides weather and crowd forecasts.  

4. **Assistant Companion System**  
   - **Velora**: Emotional support chatbot for stress management.  
   - **Zara**: Cultural guide offering insights into Sri Lanka’s heritage.  
   - **Einstein (MathBot)**: Handles mathematical queries and unit conversions.   

---


## **Contributors**
- **Team Members**: Nadara, Dananji, Shashika, and Tharaka 
- **Supervisors**: Ms. Thilini Jayalath and Dr. Lakmini Abeywardhana

---
---


# **The Companion System: Emotional and Cultural Assistance for SL Voyager: IT20391768**

---

## **Overview**  
The Companion System is a key component of **SL Voyager**, designed to enrich the travel experience by providing tourists with emotional support and cultural insights. It includes two major chatbots:  
1. **Velora**: An emotional support assistant for stress management.  
2. **Zara**: A cultural guide that helps tourists connect deeply with Sri Lanka’s heritage.  

Additionally, it features Einstein, the MathBot, for practical assistance with unit conversions and basic calculations.  

---

## **Features**  

### **Velora**: Emotional Support Chatbot  
- Provides stress-relieving advice in a conversational and comforting manner.  
- Redirects users to other SL Voyager systems for attractions, hotels, or forecasting.  
- Engages users with jokes, lighthearted conversation, and empathy.  

### **Zara**: Cultural Guide  
- Offers rare and valuable cultural insights into Sri Lanka’s traditions and heritage.  
- Helps tourists feel more connected to the local culture.  

### **Einstein**: MathBot  
- Handles mathematical queries, including unit conversions (e.g., metric to imperial).  
- Helps users adapt to local measurements effortlessly.  

---

## **Installation**  
1. Clone the repository

2. Install dependencies

3. Run the Flask server:  
   ```bash
   python index.py
   ```  
4. Access the system in your browser:  
   - Velora: `http://127.0.0.1:5000/`  
   - MathBot: `http://127.0.0.1:5000/mathbot`  

---

## **Technology Stack**  
- **Frontend**: HTML, CSS (custom themes for emotional and romantic vibes)  
- **Backend**: Python (Flask, ChatterBot)  
- **Libraries Used**:  
  - ChatterBot for chatbot logic  
  - Pint for unit conversions  

---

## **Usage**  
1. Access Velora to:  
   - Get emotional support during stressful travel situations.  
   - Redirect queries to other SL Voyager systems like attractions, hotels, or forecasting.  

2. Use Zara to:  
   - Learn about Sri Lanka’s rich culture and traditions.  

3. Use Einstein to:  
   - Perform quick math operations and unit conversions.  

---

## **Future Enhancements**  
- Zara  
- A weather bot  
- Integrate every bot into a single system
- UIs with high user experience  


**Developer**: Tharaka V B Pallevela [IT20391768]  

---



# The sub component:Attraction Suggestion System - IT21345746 

## Overview 

The Attraction Suggestion System is a recommendation engine that uses user preferences to suggest tourist attractions and events based on real time seasonal data.Along with practical travel tips that aligns with differenet weather conditions. The system is designed to make travel planning intuitive and enjoyable by leveraging machine learning models to provide personalized recommendations.And specially made for Sri Lankan tourism.




## Features
**Personalized attraction recommendation**

**Seasonal and event-based suggestions**

**Questionaries survey**

**Practical travel tips**

**Made for Sri Lanka**


## Technology Stack
**Machine Learning Frameworks**
- TensorFlow/Keras: For model training and prediction.
- Scikit-learn: For data preprocessing and encoding.
Model partially created with 81.82% accuracy

**Programming Language**
- Python

**Libraries**
- pandas for data handling.
- numpy for numerical computations

**Front-End**
- HTML/CSS
- Javascript(React)

**Dataset fields**
- Name: Attraction name.
- Description: Short description of the attraction.
- Type: Type of activity associated with the attraction.
- Season: Best season(s) for visiting the attraction.
- Location: Geographical location of the attraction.


## Installation
- Clone the repository
- Install the dependencies
- Run the Flask Server
```bash
  python app.py
```
- Access the System in your browser
`http://127.0.0.1:5000/`

## Usage
- Select one or more seasons from the available options.
- Select activity types that match the selected seasons.
- Receive recommendations for attractions, including names and locations


## Future Enhancements

- Adjust the system with upcoming events
- Recommendations according to real-time weather
- Practicle tips suggestion according to weather and the recommended places


**Developer**: Siriwardana A.P.G.D.P [IT21345746]



# Seasonal Tourism Forecasting System: IT21245060

## Overview

The Seasonal Tourism Forecasting System is a data-driven engine designed to predict tourism demand based on seasonal patterns, real-time data, and historical trends. By leveraging machine learning models, it forecasts peak and off-season periods, providing valuable insights for tourism planning. The system is tailored to improve resource allocation, enhance marketing strategies, and optimize operational planning for tourism agencies.

## Features
**Seasonal Demand Forecasting: Predicts tourism demand for various seasons based on historical and real-time data**

**Trend Prediction: Analyzes long-term trends and makes predictions about future tourism volumes**

**Real-time Data Integration: Incorporates up-to-the-minute data from weather reports, social media, and travel behavior to refine predictions**

**Optimized Resource Allocation: Helps travel agencies plan for peak and off-season resource management**

**Customizable Parameters: Allows users to adjust forecasts based on their own specific data, including regional variations and local events**

## Technology Stack
**Machine Learning Frameworks**
-TensorFlow/Keras: For model training and prediction.
-Scikit-learn: For data preprocessing and encoding.

**Programming Languag**
-Python: Primary language used for system development.

**Libraries**
-Pandas: For data handling.
-Numpy: For numerical computations.

**Front-End**
-HTML/CSS: For the system interface.
-JavaScript (React): For enhancing user interactivity.

**Dataset Fields**
-Tourism Demand: Number of visitors in a given period.
-Date/Time: Specific date and season.
-Location: Geographical location of interest (city, region, etc.).
-Weather Conditions: Data on weather forecasts impacting tourism.
-Event Data: Information about local events, holidays, and festivals.

## Installation
-Clone the repository.
-Install dependencies using:
 pip install -r requirements.txt
-Run the Flask server:
 ```
 python app.py
```
-Access the system via the browser at: 
```http://127.0.0.1:5000/```

## Usage
-Select specific seasons or time periods for forecasting.
-Input regional data and event schedules for more accurate predictions.
-Receive seasonal forecasts, including insights on peak tourism periods, visitor counts, and resource needs.

## Future Enhancements
=Real-Time Data Integration: Incorporate more dynamic, real-time sources such as social media and mobile application usage to refine predictions.
=Advanced Trend Analytics: Use deep learning models for improved accuracy and to predict longer-term trends.
=User Customization: Enable personalized forecast parameters for specific regions or tourist attractions.

**Developer**: Weerakoon H.P.S.P. [IT21245060]

## **Contributors**
-**Team Members**: Tharaka, Nadara, Dananji, Shashika
-**Supervisors**: Ms. Thilini Jayalath and Dr. Lakmini Abeywardhana

---

