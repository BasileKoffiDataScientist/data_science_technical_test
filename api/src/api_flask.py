import torch
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_restful import Resource, Api, reqparse
# from main import predict

from main import predict, main, row_for_prediction, predict_final

app = Flask(__name__)
api = Api(app)

# print('[INFO: Prediction start here!!!!]')
model_1 = torch.jit.load('../models/model_scripted_1.pt')
model_1.eval()
# print(model_1)

model_2 = torch.jit.load('../models/model_scripted_2.pt')
model_2.eval()
# print(model_2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/molecule', methods=['POST', 'GET'])
def predict():
    if request.method=='POST':
        molecule = request.form['molecule']
        output = predict_final(molecule)
        return redirect(url_for("models", pred=output))

    else:
        return render_template('prediction.html')


@app.route("/<pred>", methods=['POST', 'GET'])
def models(pred):
    return f"<h1>{pred}</h1>"


    #
    # if request.method == 'GET':
    #     my_prediction = pred
    #     return redirect(url_for('prediction.html')) # f"<h1>{pred}</h1>"




@ app.route('/single_data')
def single_data_view():
    return render_template('profile_report_data_4_model_model_1.html')


@app.route('/multi_data')
def multi_data_view():
    return render_template('profile_report_data_4_model_multi_model_3.html')

# @app.route('/', methods=['POST', 'GET'])
# def predict():
#     if request.method == 'POST':
#         molecule = request.form['molecule']#files.get('molecule')
#         return redirect(url_for())
#         if molecule is None:
#             return jsonify({'error': 'no molecule'})
#         # if not allowed_file(file.filename):
#         #     return jsonify({'error': 'format not supported'})
#
#         try:
#             lst_fp, list_smile_normalize = row_for_prediction(molecule)
#             row_fp_tensor = torch.FloatTensor(lst_fp)
#             row_smile_tensor = torch.FloatTensor(list_smile_normalize)
#             prediction_1 = predict(row_fp_tensor, model_1).argmax().item()
#             data_1 = {'prediction': prediction_1, 'class_name': str(prediction_1)}
#             prediction_2 = predict(row_smile_tensor, model_2).argmax().item()
#             data_2 = {'prediction': prediction_2, 'class_name': str(prediction_2)}
#             data_final = {
#                 'data_1': {'prediction': prediction_1, 'class_name': str(prediction_1)},
#                 'data_2': {'prediction': prediction_2, 'class_name': str(prediction_2)},
#
#             }
#             return jsonify(data_final)
#         except:
#             return jsonify({'error': 'error during prediction'})

# @app.route('/predict/model_1',methods=['POST'])
# def predict(data):
#     isinstance(data, str)
#     y_pred = main(data)
#     return {'status': 'OK', 'y_pred': y_pred.tolist()}


# @app.route('/predict/<path:path_X_test>',methods=['POST'])
# def predict(path_X_test):
#     isinstance(path_X_test, str)
#     y_pred = predict(path_X_test)
#     return {'status': 'OK', 'y_pred': y_pred.tolist()}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


# parser = reqparse.RequestParser()
class My_Molecule(Resource):
    def put(self, data):
        output = predict_final(data)
        return jsonify(output)


    # def post(self):
    #     args = parser.parse_args()
    #     data = 'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C'
    #     the_molecule = args[data]
    #
    #     data_final = predict_final(the_molecule)
    #     # # save a temporary copy of the file
    #     # lst_fp, list_smile_normalize = row_for_prediction(the_molecule)
    #     # row_fp_tensor = torch.FloatTensor(lst_fp)
    #     # row_smile_tensor = torch.FloatTensor(list_smile_normalize)
    #     # prediction_1 = predict(row_fp_tensor, model_1).argmax().item()
    #     # data_1 = {'prediction': prediction_1, 'class_name': str(prediction_1)}
    #     # prediction_2 = predict(row_smile_tensor, model_2).argmax().item()
    #     # data_2 = {'prediction': prediction_2, 'class_name': str(prediction_2)}
    #     # data_final = {
    #     #     'data_1': {'prediction': prediction_1, 'class_name': str(prediction_1)},
    #     #     'data_2': {'prediction': prediction_2, 'class_name': str(prediction_2)},
    #     #
    #     # }
    #
    #     return jsonify({'data': data_final}), 201


api.add_resource(HelloWorld, '/hello')
api.add_resource(My_Molecule, '/final/<data>')

if __name__ == '__main__':
    app.run('0.0.0.0', 5000)