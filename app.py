from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
with open('iris_model.pkl','rb') as f:
    model=pickle.load(f)
@app.route('/home', methods=['GET'])
def home():
    return "hi, I am Nandana"

@app.route('/dynamic/<var>')
def dynamicurl(var):
    return f"hi, I am {var}"


@app.route('/posthome', methods=['POST'])
def postreq():
    data=request.get_json()
    return jsonify({"data":data}),200

@app.route('/postmodel', methods=['POST'])
def predictfn():
    data=request.get_json()
    features = np.array([data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']]).reshape(1,-1)
    prediction=model.predict(features) 
    return jsonify({"data":int(prediction[0])}),200                                                                                                         
    
    


if __name__ == '__main__':
    app.run(debug=True)
    
    
### nandanasreeraj123