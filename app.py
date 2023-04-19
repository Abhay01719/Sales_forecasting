from flask import Flask, render_template, jsonify, request
import joblib
import os
import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/result')
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    x_array = np.array([[item_weight, item_fat_content,item_visibility, item_type, item_mrp,
                  outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    scaler_path = r'Sc.sav'

    sc = joblib.load(scaler_path)

    x_standard = sc.transform(x_array)

    model_path = r'Lr.sav'

    model = joblib.load(model_path)

    y_pred = model.predict(x_standard)

    return render_template('form.html', results =" The predicted value is{}".format(y_pred) )
    # ({'Prediction': float(y_pred)})
   

if __name__ == "__main__":
    app.run(debug=True, port=8000)
