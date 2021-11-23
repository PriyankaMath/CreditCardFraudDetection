from flask import Flask, render_template, url_for, request
import pandas as pd, numpy as np
import pickle

# load the model from disk
filename = 'knn_model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        me = request.form['message']
        print("inside predict", me)
        message = [float(x) for x in me.split()]
        print("after message",)
        vect = np.array(message).reshape(1, -1)
        print(vect)
        final_prediction = clf.predict(vect)
    return render_template('result.html', prediction = final_prediction)

if __name__== "__main__":
    app.run(debug=True)