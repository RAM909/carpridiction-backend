# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load machine learning models

rf_model = pickle.load(open('randomforest.pkl', 'rb'))
# ada_model = pickle.load(open('adaboost.pkl', 'rb'))
# bg_model = pickle.load(open('bagging.pkl', 'rb'))
# adt_model = pickle.load(open('adadt.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)

    # Extract input features from request data
    km = data['kmDriven']
    year = data['year']
    fuel_type = data['fuelType']
    seller_type = data['sellerType']
    ownertype = data['owner']
    transmission = data['transmission']
    manufacturer = data['manufacturer']
    option = data['selectedModel']

    if option=='AdaBoost Regressor':
     model=adt_model
    elif option=='AdaBoost Regressor 2':
     model=ada_model
    elif option=='Bagging Regressor':
     model=bg_model
    else:
     model=rf_model
     
     
    n=np.zeros([1,34],dtype=int)
    
    
   
    n[0,0]=km


   
    n[0,1]=year


    #for fuel
    fuel = fuel_type

    if fuel=='CNG':
     n[0,2]=1
    elif fuel=='Diesel':
     n[0,3]=1
    elif fuel=='Electric':
     n[0,4]=1
    elif fuel=='LPG':
     n[0,5]=1
    else:
     n[0,6]=1
    
    
    #for seller type
    s_t = seller_type
    if s_t=='Dealer':
     n[0,7]=1
    elif s_t=='Individual':
     n[0,8]=1
    else:
     n[0,9]=1
    

    #for owner
    owner = ownertype

    if owner=='First Owner':
     n[0,10]=1
    elif owner=='Fourth & Above Owner':
     n[0,11]=1
    elif owner=='Second Owner':
     n[0,12]=1
    elif owner=='Test Drive Car':
     n[0,13]=1
    else:
     n[0,14]=1
    
    
    # for transmission
    tran = transmission


    if tran=='Automatic':
     n[0,15]=1
    else:
     n[0,16]=1
    
    
    #for manufacturer
    man = manufacturer

    if man=='Audi':
     n[0,17]=1
    elif man=='BMW':
     n[0,18]=1
    elif man=='Chevrolet':
     n[0,19]=1
    elif man=='Datsun':
     n[0,20]=1
    elif man=='Fiat':
     n[0,21]=1
    elif man=='Ford':
     n[0,22]=1
    elif man=='Honda':
     n[0,23]=1
    elif man=='Hyundai':
     n[0,24]=1
    elif man=='Mahindra':
     n[0,25]=1
    elif man=='Maruti':
     n[0,26]=1
    elif man=='Mercedes-Benz':
     n[0,27]=1
    elif man=='Nissan':
     n[0,28]=1
    elif man=='Renault':
     n[0,29]=1
    elif man=='Skoda':
     n[0,30]=1
    elif man=='Tata':
     n[0,31]=1
    elif man=='Toyota':
     n[0,32]=1
    else:
     n[0,33]=1
     
    data=pd.DataFrame(n)
     
    result = model.predict(data)


    return jsonify({'prediction':result[0]})


if __name__ == '__main__':
    app.run(debug=True)
