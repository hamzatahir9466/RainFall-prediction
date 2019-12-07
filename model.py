import pandas as pd
import numpy as np
import  sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import requests
import json

def linearRegression ():
	# Read The data in a pandas dataframe
	data = pd.read_csv("austin_weather.csv")

	# Drop or Delete the unnecessary columns in the data.
	data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches',  
	                  'SeaLevelPressureLowInches'], axis = 1) 

	# Some values of 'T' which denotes trace rainfall
	# We need to replace all occurrence of T with 0
	# So that we can use the data in our model
	data = data.replace('T', 0.0)

	# The data also contains '-' which indicates no
	# or NIL. This means that data is not available
	# we need to replace these values as well.
	data = data.replace('-', 0.0)

	# Save the data in a CSV File
	data.to_csv('austin_final.csv')

	# Linear Regression is a linear approach to form a 
	# relationship between dependent variable and many 
	# independent explanatory variables



	# Read the cleaned data 
	data = pd.read_csv("austin_final.csv")
	# stuff
	X = data.drop(['PrecipitationSumInches'], axis = 1)

	# the output or the label. 
	Y = data['PrecipitationSumInches'] 
	# reshaping it into a 2-D vector 
	Y = Y.values.reshape(-1, 1) 

	day_index = 798
	days = [i for i in range(Y.size)] 

	clf = LinearRegression() 
	clf.fit(X, Y) 

	pickle.dump(clf, open('machinelearning.pkl', 'wb'))
	pickle.dump(days, open('days.pkl', 'wb'))
	pickle.dump(Y, open('Y.pkl', 'wb'))
	pickle.dump(X, open('X.pkl', 'wb'))

	# model = pickle.load(open('machinelearning.pkl', 'rb'))
	# model2 = pickle.load(open('days.pkl', 'rb'))
	# model3 = pickle.load(open('Y.pkl'), 'rb')


	inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45], 
	                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]]) 
	inp = inp.reshape(1, -1) 


