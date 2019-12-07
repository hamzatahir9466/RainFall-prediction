import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from .model import linearRegression 
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Run the Model
linearRegression()

# Load the model
model  = pickle.load(open('model.pkl','rb'))
model2 = pickle.load(open('days.pkl', 'rb'))
model3 = pickle.load(open('Y.pkl', 'rb'))
model4 = pickle.load(open('X.pkl', 'rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.

    data = request.get_json(force=True)
    
  
    inp = np.array([data['daydata'][0],data['daydata'][1],data['daydata'][2], data['daydata'][3], data['daydata'][4], data['daydata'][5],
                                            data['daydata'][6], data['daydata'][7], data['daydata'][8], data['daydata'][9], data['daydata'][10],
                                            data['daydata'][11], data['daydata'][12], data['daydata'][13], data['daydata'][14], data['daydata'][15],data['daydata'][16]])
    inp = inp.reshape(1, -1)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(inp).tolist()

    days = model2
   # print(days, type(days))
    Y    = model3.tolist()
    
    Y_cleaned = []
    i = 0
    for  i  in range(len(Y) - 1):
        Y_cleaned.append(Y[i][0])

    # Take the first value of prediction
    output = prediction

    d_vs_y = [days, Y]

    xdatabase = model4

    exportdataframe = xdatabase.to_json()
    #print(output)
    return jsonify(daydata = days, ydata = Y_cleaned, predictiondata = output, xdata=exportdataframe)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
    	print("Server is exited unexpectedly. Please contact server admin.")
