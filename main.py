import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from db import users_collection
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load ML models
classifier = joblib.load('crop_rm.joblib')
data = pickle.load(open('Label.pkl', 'rb'))

# Configure Gemini API (use environment variable for security)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------- ROUTES -------------------------

# Login Page
@app.route('/')
@app.route('/login')
def login():
    return render_template('login.html')

# Login Request Handler
@app.route('/lr', methods=['GET', 'POST'])
def loginrequest():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = users_collection.find_one({'username': username, 'password': password})
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('invalid_login.html')
    return render_template('login.html')

# Signup Page
@app.route('/signup')
def singup():
    return render_template('signup.html')

# Registration Handler
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            return render_template('user_already.html')
        else:
            users_collection.insert_one({
                'name': name,
                'email': email,
                'username': username,
                'password': password
            })
            flash('Signup successful! Please log in.')
            return render_template('valid_signup.html')

# Home Page
@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

# Dashboard
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return f"Hello {session['username']}, welcome to your dashboard!"
    return redirect(url_for('login'))

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Modules
@app.route('/home/CRS')
def crs():
    return render_template('crop_recommendation.html')

@app.route('/home/disease_detect')
def dds():
    return render_template('disease_detection.html')

@app.route('/home/weather_forecast')
def wfs():
    return render_template('weather_forcast.html')  # you can show "Service Not Available" on this template

@app.route('/home/soil_test')
def sts():
    return render_template('soil_test.html')

# --------------------------------------------
# Crop Recommendation Prediction Route
# --------------------------------------------
@app.route('/CRS/Predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form input
            n = int(request.form.get('N'))
            p = int(request.form.get('P'))
            k = int(request.form.get('K'))
            ph = float(request.form.get('PH'))
            t = float(request.form.get('T'))
            h = float(request.form.get('H'))
            r = float(request.form.get('R'))

            # Create DataFrame
            input_df = pd.DataFrame({
                'N': [n], 'P': [p], 'K': [k],
                'temperature': [t], 'humidity': [h],
                'ph': [ph], 'rainfall': [r]
            })

            # Predict crop label
            result = classifier.predict(input_df)[0]
            final_result = data.index[data.Label == result][0]

            # Background color mapping for crops
            bg_map = {
                'rice': 'from-lime-100 to-green-300',
                'maize': 'from-yellow-200 to-amber-300',
                'chickpea': 'from-yellow-100 to-stone-200',
                'kidneybeans': 'from-red-200 to-rose-300',
                'pomegranate': 'from-pink-200 to-red-400',
                'banana': 'from-yellow-200 to-yellow-400',
                'mango': 'from-amber-200 to-orange-300',
                'grapes': 'from-purple-200 to-violet-400',
                'watermelon': 'from-pink-200 to-green-300',
                'muskmelon': 'from-yellow-200 to-orange-300',
                'apple': 'from-green-200 to-red-300',
                'orange': 'from-orange-200 to-orange-400',
                'papaya': 'from-orange-200 to-yellow-300',
                'coconut': 'from-emerald-200 to-brown-300',
                'cotton': 'from-gray-100 to-white',
                'jute': 'from-yellow-100 to-green-200',
                'coffee': 'from-brown-200 to-stone-400',
                'lentil': 'from-orange-100 to-amber-200',
                'mungbean': 'from-green-200 to-emerald-400',
                'mothbeans': 'from-yellow-200 to-amber-300',
                'pigeonpeas': 'from-yellow-200 to-orange-300',
                'blackgram': 'from-gray-200 to-zinc-400'
            }
            bg_class = bg_map.get(final_result.lower(), 'from-sky-100 to-emerald-100')

            # Gemini Prompt
            prompt = (
                f"Based on the conditions: Nitrogen={n}, Phosphorus={p}, Potassium={k}, "
                f"Temperature={t}°C, Humidity={h}%, pH={ph}, Rainfall={r}mm. "
                f"The predicted crop is {final_result}. "
                f"Return the following fields in valid JSON format:\n\n"
                "{\n"
                "  \"practical_tips\": [\"...\", \"...\"],\n"
                "  \"ideal_conditions\": {\n"
                "    \"temperature\": \"...\",\n"
                "    \"humidity\": \"...\",\n"
                "    \"soil_ph\": \"...\"\n"
                "  },\n"
                "  \"market_insights\": \"...\"\n"
                "}\n\n"
                "Important: Return only the JSON. No explanation, no markdown, no labels — just the JSON."
            )

            # Gemini Call
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            if not raw_text:
                raise ValueError("Gemini returned an empty response.")

            # Parse JSON
            try:
                insights = json.loads(raw_text)
            except json.JSONDecodeError:
                if raw_text.startswith("```json"):
                    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
                    insights = json.loads(raw_text)
                else:
                    raise


            return render_template("crs_result.html", result=final_result, insights=insights, bg_class=bg_class)

        except Exception as e:
            print(f"❌ Error generating insights: {e}")
            return render_template("crs_result.html", result="Unknown", insights={
                "practical_tips": ["Something went wrong while generating insights."],
                "ideal_conditions": {
                    "temperature": "N/A",
                    "humidity": "N/A",
                    "soil_ph": "N/A"
                },
                "market_insights": "Failed to retrieve structured market information."
            })

    return "Invalid request method", 405

# ------------------------- MAIN ENTRY POINT -------------------------
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 10000))
    serve(app, host="0.0.0.0", port=port)
