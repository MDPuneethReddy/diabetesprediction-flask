from flask import Flask, render_template, request, redirect
import pickle
import numpy as np 
import pandas as pd 

model=pickle.load(open("diabetes.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    float_feat=[float(x) for x in request.form.values()]
    finalarr=[np.array(float_feat)]
    prediction=model.predict(finalarr)
    if prediction==1:
        pred="YOU HAVE DIABETES"
    elif prediction==0:
        pred="NO YOU DONT HAVE DIABETES"
    output=pred
    return render_template('result.html',pred_text='{}'.format(output))

@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')
    

if __name__ == "__main__":
    app.run(debug=True)
