from flask import Flask,request
import pickle
import numpy as np
import sklearn
from logging import FileHandler,WARNING
app = Flask(__name__, template_folder = 'template')

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
def hello_world():
    return 'Hello World'

"""
{
  "pH":7,
  "HCO3":45,
  "PCO2":29
}
"""
@app.route('/Mpredict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    pickled_model = pickle.load(open('Rest_Api_MLmodel/model_pkl.pkl', 'rb'))
    print(data)
    print(data['pH'],data['HCO3'],data['PCO2'])
    temp_test = np.reshape([data['pH'],data['HCO3'],data['PCO2']], (1, -1))
    temp = pickled_model.predict(temp_test)
    return temp[0]

@app.route('/PPpredict', methods=['GET', 'POST'])
def predict_p():
    data1 = request.get_json()
    pickled_model1 = pickle.load(open('Rest_Api_MLmodel/UClassifier.pkl', 'rb'))
    print(data1)
    print(data1['pH'],data1['HCO3'],data1['PCO2'])
    temp_test = np.reshape([data1['pH'],data1['HCO3'],data1['PCO2']], (1, -1))
    temp = pickled_model1.predict(temp_test)
    return temp[0]

@app.route('/Spredict', methods=['GET', 'POST'])
def predict_sm():
    data1 = request.get_json()
    pickled_model1 = pickle.load(open('spiro_pkl.pkl', 'rb'))
    print(data1)
    print(data1['pH'],data1['HCO3'],data1['PCO2'])
    temp_test = np.reshape([data1['FVC'],data1['FEV1'],data1['AGE'],data1["gender"]], (1, -1))
    temp = pickled_model1.predict(temp_test)
    return temp[0]



##
"""""
{
    "Na": 166,
    "k": 5.4,
    "Cl": 111,
    "Ca": 1.06,
}
"""
@app.route('/Electrolyte', methods=['GET', 'POST'])
def Electrolyte_A():
    Ans = ""
    data = request.get_json()
    #Soduim
    if data['Na'] > 143:
        Ans += "Hypernatremia and " 
    elif data['Na'] < 135:
        Ans += "Hyponatremia and "
    #Potasuim
    if data['k'] > 4.5:
        Ans += "Hyperkalemia and "
    elif data['k'] < 3.5:
        Ans += "Hypokalemia and "
    #Calcuim
    if data['Ca'] > 1.27:
        Ans += "Hypercalcemia and "
    elif data['Ca'] < 1.17:
        Ans += "Hypocalcemia and "
    #Chlorine
    if data['Cl'] > 107:
        Ans += "Hyperchloremia"
    elif data['Cl'] < 96:
        Ans += "Hypochloremia"
    return Ans

# main driver function
if __name__ == '__main__':
    app.run()