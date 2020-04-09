# -*- coding: utf-8 -*-

import json
from ml_evaluator import get_model_instances, get_instance_runs, register_model
from ml_evaluator import get_user_models
from ml_evaluator import MLEvaluator
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to MLEvaluator -- use this service to test model instances"

@app.route('/models/<user_name>/', methods=['GET'])
def get_models(user_name):
    return jsonify(get_user_models(user_name))

@app.route('/instances/<user_name>/<model_name>/', methods=['GET'])
def get_instances(user_name, model_name):
    return jsonify(get_model_instances(user_name, model_name))

@app.route('/runs/<user_name>/<model_name>/<instance_name>')
def get_runs(user_name, model_name, instance_name):
    return jsonify(get_instance_runs(user_name, model_name, instance_name))

@app.route('/evaluator/kfold', methods=['POST'])
def run_k_crossfold_validation():
    req_data = request.get_json()

    user_name = req_data['user_name']
    model_name = req_data['model_name']

    register_model(user_name, model_name)

    instance_name = ['instance_name']
    hyperparams = req_data['hyperparams']

    model_s3_key = req_data['model_s3_key']
    train_data_s3_key = req_data['train_data_s3_key']
    test_data_s3_key = req_data['test_data_s3_key']

    mle = MLEvaluator(model_name=model_name,
                      instance_name=instance_name,
                      model_s3_key=model_s3_key,
                      train_data_s3_key=train_data_s3_key,
                      test_data_s3_key=test_data_s3_key)

    k = req_data['k']
    shuffle = req_data.get('shuffle', True)
    random_gen = req_data.get('random_gen', 1)

    mle.run_k_crossfold_validation(k=k, shuffle=True, random_gen=1)

    return jsonify(get_instance_runs(user_name, model_name, instance_name))

if __name__ == '__main__':
    app.run()
