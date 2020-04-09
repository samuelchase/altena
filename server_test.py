# -*- coding: utf-8 -*-

import requests

print('checking...  /')
response = requests.get('http://127.0.0.1:5000/')
print(response.content)

print('initializing ML model')

d = {
     'user_name': 'sam',
     'model_name': 'county_classifier',
     'instance_name': 'non_china_500_32'
     'hyperparams': [500, 32, 32]
     'model_s3_key': 'county_classifier.pkl'
     'train_data_s3_key': 'ds-project-test.csv'
     'test_data_s3_key': 'ds-project-train.csv'
     'k': 10
}


response = requests.post('http://127.0.0.1:5000/evaluator/kfold', json=d)
print(response.content)

print('\nget models')
response = requests.get('http://127.0.0.1:5000/models/sam/')
print(response.content)

print('\nget instances')
response = requests.get('http://127.0.0.1:5000/instances/sam/county_classifier')
print(response.content)

print('\nget runs')
response = requests.get('http://127.0.0.1:5000/runs/sam/county_classifier/non_china_500_32')
print(response.content)