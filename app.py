
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

def predictor(input_params):
    re_data = np.array(input_params).reshape(1, 4)
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(re_data)
    return prediction[0]
 
app = Flask(__name__)

# hi.com = hi.com/

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        list_input_params = request.form.to_dict()
        # request.form => immutableDict[('name':value)]
        # {'name':value}
        list_input_params = list(list_input_params.values())
        list_input_params = list(map(float, list_input_params))
        # ["2", "3", "4", "5"]
        prediction = predictor(list_input_params)
        # return jsonify(prediction)
        return render_template('predict.html', finale=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)
