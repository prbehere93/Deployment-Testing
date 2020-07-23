import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pandas as pd

app = Flask(__name__)
#transform=pickle.load(open('transform.pkl','rb'))
#model = pickle.load(open('model.pkl', 'rb'))	
t1=pickle.load(open('wl.pkl','rb')) #WordNetLemmatizer
t2=pickle.load(open('tfidf.pkl','rb')) #TFIDF transformer
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():	
	if request.method=='POST':
		message = request.form["message"]
		data = [message]
		data=pd.Series(data)
		data=data.apply(t1.lemmatize)
		data=t2.transform(data)
		prediction = model.predict(data)
	return render_template('index.html', prediction_text=str('The News is '+str(prediction[0])))


if __name__ == "__main__":
	app.run(debug=True)
