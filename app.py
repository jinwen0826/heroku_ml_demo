import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    features_list = [float(x) for x in request.form.values()]
    features = np.array(features_list).reshape(1,-1)
    predict_outcome_list = model.predict(features)
    predict_outcome = round(predict_outcome_list[0],2)

    return render_template('page.html',prediction_display_area='Predicted Price is ：{}'.format(predict_outcome))

if __name__ == "__main__":
    app.run(port=5000,debug = True)
