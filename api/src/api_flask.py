from flask import Flask, jsonify, request, render_template
# from main import predict

from main import predict, main

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/single_data')
def single_data_view():
    return render_template('profile_report_data_4_model_model_1.html')


@app.route('/multi_data')
def multi_data_view():
    return render_template('profile_report_data_4_model_multi_model_3.html')


@app.route('/predict/model_1',methods=['POST'])
def predict(data):
    isinstance(data, str)
    y_pred = main(data)
    return {'status': 'OK', 'y_pred': y_pred.tolist()}


# @app.route('/predict/<path:path_X_test>',methods=['POST'])
# def predict(path_X_test):
#     isinstance(path_X_test, str)
#     y_pred = predict(path_X_test)
#     return {'status': 'OK', 'y_pred': y_pred.tolist()}


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)