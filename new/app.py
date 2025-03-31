from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('Fertclassifier.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    Moisture = request.form['Moisture']
    soil_type = request.form['soil_type']
    crop_type = request.form['crop_type']

    soil_dict={
        'Loamy':1,
        'Sandy':2,
        'Clayey':3,
        'Black':4,
        'Red':5
    }

    crop_dict={
        'Sugarcane':1,
        'Cotton':2,
        'Millets':3,
        'Paddy':4,
        'Pulses':5,
        'Wheat':6,
        'Tobacco':7,
        'Barley':8,
        'Oil seeds':9,
        'Ground Nuts':10,
        'Maize':11
    
    }
    print(soil_dict['Red'])
    feature_list = [temp, humidity, Moisture, N, K, P, soil_dict[soil_type.split('\t')[0]], crop_dict[crop_type.split('\t')[0]]]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)
    result = "{} is the best crop to be cultivated right there".format(prediction)
    # else:
    #     result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)