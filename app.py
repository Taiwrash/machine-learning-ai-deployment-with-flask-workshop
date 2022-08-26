from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

def predictor(input_params):
    to_predict = np.array(input_params).reshape(1,4)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
 
app = Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods = ["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = predictor(to_predict_list)
        prediction = str(result)
    return render_template("predict.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
