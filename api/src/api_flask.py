from flask import Flask, jsonify, request, render_template
from main import Predict

from main import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/<path:path_X_test>')
def predict(path_X_test):
    """

    """
    isinstance(path_X_test, str)
    y_pred = Predict(path_X_test)
    return {'status': 'OK', 'y_pred': y_pred.tolist()}


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)