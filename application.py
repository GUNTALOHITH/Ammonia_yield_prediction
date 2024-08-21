import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        Pressure = float(request.form.get('Pressure'))
        Catalyst_Concentration = float(request.form.get('Catalyst_Concentration'))
        Feed_Composition = float(request.form.get('Feed_Composition'))
        Reaction_Time = float(request.form.get('Reaction_Time'))
        Reactor_Volume = float(request.form.get('Reactor_Volume'))
        Cooling_Rate = float(request.form.get('Cooling_Rate'))
        Agitation_Speed = float(request.form.get('Agitation_Speed'))
    
        new_data_scaled=standard_scaler.transform([[Temperature,Pressure,Catalyst_Concentration,Feed_Composition,Reaction_Time,Reactor_Volume,Cooling_Rate,Agitation_Speed]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0" , port= 8000)
