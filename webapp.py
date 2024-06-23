from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
scaler = joblib.load(open('scaler.pkl', 'rb'))
model = joblib.load(open('best_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the json request

    user_input = {}
    user_input["potential"] = request.get_json()['potential']
    user_input["passing"] = request.get_json()['passing']
    user_input["dribbling"] = request.get_json()['dribbling']
    user_input["attacking_short_passing"] = request.get_json()['attacking_short_passing']
    user_input["skill_long_passing"] = request.get_json()['skill_long_passing']
    user_input["movement_reactions"] = request.get_json()['movement_reactions']
    user_input["power_shot_power"] = request.get_json()['power_shot_power']

    # make prediction

    user_input_dataframe = pd.DataFrame([user_input])
    user_input_dataframe = scaler.transform(user_input_dataframe)


    prediction = model.predict(user_input_dataframe)
    int_prediction = int(round(prediction[0], 0))


    return jsonify({'prediction': int_prediction})

if __name__ == '__main__':
    app.run(debug=True)