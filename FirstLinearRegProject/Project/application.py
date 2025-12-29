from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load trained model & scaler
ridge_model = pickle.load(open('./models/ridge_regressor_model.pkl', 'rb'))
standard_scaler = pickle.load(open('./models/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        input_data = np.array([[ 
            Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
        ]])

        scaled_data = standard_scaler.transform(input_data)
        prediction = ridge_model.predict(scaled_data)[0]

        return render_template('home.html', results=round(prediction, 2))

    except Exception as e:
        return render_template('home.html', results="Invalid input!")


if __name__ == '__main__':
    app.run(port=5001, debug=True)
