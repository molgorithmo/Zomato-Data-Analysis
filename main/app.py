import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
le = pickle.load(open('label.pkl','rb'))
ohe = pickle.load(open('hot.pkl','rb'))

@app.route('/')
def home():
    return render_template('zom.html')

@app.route('/predict',methods=['POST'])
def predict():
    met_df = pd.read_csv('../datasets/cleaned_dataset.csv', index_col=0)

    int_features = [[x for x in request.form.values()]]
    node = pd.DataFrame(int_features)
    node.columns = met_df.drop('cost_for_two',axis=1).columns
    val = le.transform(node['rest_type']).reshape(-1,1)
    res = node['rest_type'].values
    res = ohe.transform(val)
    res = pd.DataFrame(res)
    new = pd.concat([node,res], axis=1)
    new.dropna(inplace=True)
    new.drop('rest_type', axis=1, inplace=True)

    prediction = model.predict(new)

    output = round(prediction[0], 2)

    return render_template('zom.html', prediction_text='Cost for two should be $ {}'.format(output))

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)