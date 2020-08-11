import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.preprocessing import text, sequence
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

max_features = 10000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():	
	if request.method=='POST':
		message = request.form["message"]
		data = [message]
		tokenizer.fit_on_texts(data)
		tokenized_train = tokenizer.texts_to_sequences(data)
		data = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
		prediction = model.predict_classes(data)
		if prediction==1:
			prediction="Real"
		else:
			prediction="Fake"	
	return render_template('index.html', prediction_text=str('The News is '+str(prediction)))


if __name__ == "__main__":
	app.run(debug=True)
