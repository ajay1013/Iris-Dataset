# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:44:54 2020

@author: Ajay Kumar
"""


from flask import Flask, request
import pandas as pd
import pickle
from flasgger import Swagger
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
Swagger(app)

pickle_in=open("classifier_iris.pkl", "rb")
classifier=pickle.load(pickle_in)

@app.route("/")
def welcome():
    return  "welcome everyone"

    
@app.route("/predict", methods=["GET"])
def iris_authentication():
    
    """Let's Authenticate the Flower Features
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: SepalLength
        in: query
        type: number
        required: true
        
      - name: SepalWidth
        in: query
        type: number
        required: true
        
      - name: PetalLength
        in: query
        type: number
        required: true
        
      - name: PetalWidth
        in: query
        type: number
        required: true
        
    responses:
        200:
            description: The output values
        
    """
    
    SepalLength=request.args.get("SepalLength")
    SepalWidth=request.args.get("SepalWidth")
    PetalLength=request.args.get("PetalLength")
    PetalWidth=request.args.get("PetalWidth")
    
    SepalLength=float(SepalLength)
    SepalWidth=float(SepalWidth)
    PetalLength=float(PetalLength)
    PetalWidth=float(PetalWidth)
    
    SepalLength_value = ((SepalLength)-5.843333)/(0.828066)
    SepalWidth_value = ((SepalWidth)-3.057333)/(0.328414)
    PetalLength_value = ((PetalLength)-3.758000)/(-1.397064)
    PetalWidth_value = ((PetalWidth)-1.199333)/(-1.315444)
    
    prediction=classifier.predict([[SepalLength_value, SepalWidth_value, PetalLength_value, PetalWidth_value]])
    print(prediction)
    return "Hello The answer is " + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Flower Features
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    scaler = StandardScaler()
    scaler.fit(df_test)
    df_new = scaler.transform(df_test)
    df_new = pd.DataFrame(df_new, columns=df_test.columns)
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))


if __name__=="__main__":
    app.run()